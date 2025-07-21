import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from app.image_utils import generate_recipe_image
from app.string_utils import process_json_from_recipe


class ImageRecipeAgent(BaseAgent):
    """
    A non-LLM agent that takes a recipe title and description
    and generates a hero image for it.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logging.info(f"[{self.name}] Starting image generation process.")

        recipe_summary_prompt = ctx.session.state.get("recipe_summary_prompt")

        if not recipe_summary_prompt:
            logging.warning(
                f"[{self.name}] 'recipe_summary_prompt' is missing. Skipping."
            )
            yield Event(author=self.name)
            return

        logging.info(f"[{self.name}] Generating image for: {recipe_summary_prompt}")
        image_b64 = generate_recipe_image(recipe_summary_prompt, recipe_summary_prompt)

        if image_b64:
            logging.info(
                f"[{self.name}] Successfully generated image for {recipe_summary_prompt}."
            )
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"final_recipe_image": image_b64}
                ),
            )
        else:
            logging.warning(
                f"[{self.name}] Could not generate image for {recipe_summary_prompt}. Replacing with text."
            )
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"final_recipe_image": None}
                ),
            )