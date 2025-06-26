import datetime
import logging
from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.genai import types as genai_types
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from .config import config


# --- Structured Output Models ---
class Recipe(BaseModel):
    """Model representing a complete baby food recipe."""

    title: str = Field(description="The creative and appealing name of the recipe.")
    description: str = Field(
        description="A brief, one-sentence summary of the recipe."
    )
    ingredients: list[str] = Field(
        description="A list of all ingredients with precise measurements."
    )
    instructions: list[str] = Field(
        description="Step-by-step preparation and cooking instructions."
    )
    age_appropriateness: str = Field(
        description="The recommended baby age range for this recipe (e.g., '6-8 months')."
    )


class FollowUpQuestion(BaseModel):
    """A specific question to guide recipe improvement using web search."""

    question: str = Field(
        description="A targeted question for web search to resolve a specific issue with the recipe."
    )


class PediatricianFeedback(BaseModel):
    """Model for providing evaluation feedback on a baby food recipe."""

    grade: Literal["pass", "fail"] = Field(
        description=(
            "Evaluation result. 'pass' if the recipe is safe, nutritious, and appropriate. "
            "'fail' if it requires changes."
        )
    )
    comment: str = Field(
        description=(
            "Detailed explanation of the evaluation, focusing on safety (e.g., choking hazards), "
            "nutritional balance, and age-appropriateness. Provide clear reasons for the grade."
        )
    )
    follow_up_questions: list[FollowUpQuestion] | None = Field(
        default=None,
        description=(
            "A list of specific questions to ask a search engine to fix the recipe's issues. "
            "This should be null or empty if the grade is 'pass'."
        ),
    )


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks the pediatrician's evaluation and escalates to stop the loop if the recipe passes."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("pediatrician_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Recipe evaluation passed. Escalating to stop loop."
            )
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Recipe evaluation failed or not found. Loop will continue."
            )
            yield Event(author=self.name)


# --- AGENT DEFINITIONS ---

recipe_generator = LlmAgent(
    model=config.worker_model,
    name="recipe_generator",
    description="Generates a creative and simple toddler food recipe for a child aged 1-2 years, either from a list of ingredients or from a direct recipe name.",
    instruction="""
    You are a creative chef specializing in recipes for toddlers. Your task is to create a simple, single-serving recipe suitable for a child aged 1-2 years.

    You will receive one of two inputs:
    1.  A list of one or more ingredients.
    2.  The name of a specific recipe (e.g., "Mini Chicken Meatballs").

    **TASK:**
    - If given a list of ingredients, invent a creative and simple recipe using them.
    - If given the name of a recipe, provide a simple version of that recipe.
    - All recipes should be tailored for a toddler aged 1-2 years, focusing on soft textures, small pieces, and avoiding common choking hazards.

    **RULES:**
    1.  Your output MUST be a valid JSON object that conforms to the `Recipe` schema.
    2.  The recipe must be simple, with clear instructions suitable for a beginner cook.
    3.  The `age_range` field in your output MUST be set to "1-2 years".
    """,
    output_schema=Recipe,
    output_key="current_recipe",
)

pediatrician_critic_agent = LlmAgent(
    model=config.critic_model,
    name="pediatrician_critic_agent",
    description="Evaluates a baby food recipe for safety, nutritional value, and age-appropriateness.",
    instruction=f"""
    You are a board-certified pediatrician and infant nutrition specialist.
    Your sole task is to critically evaluate the provided baby food recipe from the 'current_recipe' state key.

    **EVALUATION CRITERIA:**
    1.  **Safety:** Are the ingredients safe for babies? Is the texture and preparation method appropriate to prevent choking hazards for the specified age? (e.g., no honey for infants under 1, grapes must be quartered).
    2.  **Nutritional Value:** Is the recipe nutritionally balanced? Does it provide key nutrients for a baby's development?
    3.  **Age Appropriateness:** Is the suggested age range accurate for the ingredients and texture?

    **OUTPUT:**
    - If the recipe is excellent, grade it as "pass".
    - If you find ANY issues, you MUST grade it as "fail". Provide a detailed `comment` explaining the exact problems.
    - If the grade is "fail", you MUST provide a list of 2-3 specific `follow_up_questions` for a search engine that would help fix the problems you identified. For example: "How to safely prepare sweet potatoes for a 6 month old baby?" or "Nutritional benefits of avocado in infant diet".

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Your response must be a single, raw JSON object validating against the 'PediatricianFeedback' schema.
    """,
    output_schema=PediatricianFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="pediatrician_evaluation",
)

recipe_refiner_agent = LlmAgent(
    model=config.worker_model,
    name="recipe_refiner_agent",
    description="Refines and improves a baby food recipe based on pediatrician feedback and web research.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a recipe developer tasked with fixing a baby food recipe that failed a pediatric review.

    1.  **Review Feedback:** Carefully read the `comment` in the 'pediatrician_evaluation' state key to understand what needs to be fixed.
    2.  **Research Solutions:** Execute EVERY search query provided in the `follow_up_questions` using the `Google Search` tool to find solutions.
    3.  **Revise the Recipe:** Using the research findings, rewrite the original recipe from 'current_recipe' to address all the pediatrician's concerns. This may involve changing ingredients, measurements, or instructions.
    4.  **Output:** Your output MUST be a single, raw JSON string that strictly conforms to the `Recipe` schema. Do NOT add any markdown formatting like ```json ... ``` around the JSON string.
        The `Recipe` schema is as follows:
        {{
            "title": "string (The creative and appealing name of the recipe.)",
            "description": "string (A brief, one-sentence summary of the recipe.)",
            "ingredients": "list[string] (A list of all ingredients with precise measurements.)",
            "instructions": "list[string] (Step-by-step preparation and cooking instructions.)",
            "age_appropriateness": "string (The recommended baby age range for this recipe (e.g., '6-8 months').)"
        }}
    Ensure your entire output is ONLY this JSON string.
    """,
    tools=[google_search],
    # output_schema=Recipe,  # Removed to allow use of tools
    output_key="current_recipe",
)


final_recipe_presenter_agent = LlmAgent(
    model=config.critic_model,
    name="final_recipe_presenter_agent",
    include_contents="none",
    description="Formats the final, approved recipe into a detailed, user-friendly markdown report.",
    instruction="""
    You are a food blogger who specializes in creating beautiful and informative recipe cards for parents.
    Your task is to take the final, approved recipe data from the 'current_recipe' state key and format it into a clear and appealing markdown report.

    **REPORT STRUCTURE:**
    -   Start with the recipe `title` as a main heading (`#`).
    -   Include the `description` and `age_appropriateness`.
    -   Use a sub-heading (`##`) for "Ingredients" and list them.
    -   Use a sub-heading (`##`) for "Instructions" and list the steps.
    -   Add a "Nutrition Notes" section (`##`) with a brief, helpful summary of the recipe's health benefits for a baby.
    -   Add a "Safety First!" section (`##`) with a bullet point reminding parents to ensure the texture is appropriate for their baby's age to prevent choking.

    Your output should be a single, well-formatted markdown document.
    """,
    output_key="final_recipe_report",
)

recipe_creation_pipeline = SequentialAgent(
    name="recipe_creation_pipeline",
    description="Takes an initial recipe, runs it through an iterative refinement loop with a pediatrician critic, and then formats the final, approved recipe.",
    sub_agents=[
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                pediatrician_critic_agent,
                EscalationChecker(name="escalation_checker"),
                recipe_refiner_agent,
            ],
        ),
        final_recipe_presenter_agent,
    ],
)

interactive_recipe_agent = LlmAgent(
    name="interactive_recipe_agent",
    model=config.worker_model,
    description=(
        "The primary assistant for creating baby food recipes. It collaborates"
        " with the user to generate a recipe and then gets it approved before"
        " finalization."
    ),
    instruction=f"""
    You are a friendly and helpful AI assistant for parents, named 'Tiny Tastes'.
    Your job is to help users create delicious and healthy baby food recipes.

    **Workflow:**
    1.  **Generate Recipe:** When the user gives you a list of ingredients, your *only* first step is to call the `recipe_generator` tool.
    2.  **Present for Approval:** After the `recipe_generator` tool has run, its output will be in the 'current_recipe' state. Present this recipe to the user for approval. For example: "I've come up with a recipe called '{{current_recipe.title}}'. Does this look good? If so, I can have our AI pediatrician review it."
    3.  **Execute Refinement:** Once the user gives EXPLICIT approval (e.g., "looks good", "yes please"), you MUST delegate the task to the `recipe_creation_pipeline` agent.

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Do not perform any research or evaluation yourself. Your job is to Generate, Present, and Delegate.
    """,
    sub_agents=[recipe_creation_pipeline],
    tools=[AgentTool(recipe_generator)],
    output_key="initial_recipe",
)

root_agent = interactive_recipe_agent