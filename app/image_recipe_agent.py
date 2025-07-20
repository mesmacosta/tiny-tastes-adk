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

        raw_string = ctx.session.state.get("current_recipe")

        if not raw_string or not isinstance(raw_string, str):
            logging.warning(
                f"[{self.name}] 'current_recipe' is missing or not a string. Skipping."
            )
            yield Event(author=self.name)
            return

        recipe_data = process_json_from_recipe(raw_string)

        recipe_title = recipe_data.get("title")
        recipe_description = recipe_data.get(
            "description"
        )

        if not recipe_title or not recipe_description:
            logging.warning(
                f"[{self.name}] No 'recipe_title' or 'recipe_description' found in state. Skipping."
            )
            yield Event(author=self.name)
            return

        logging.info(f"[{self.name}] Generating image for: {recipe_title}")
        image_b64 = generate_recipe_image(recipe_title, recipe_description)

        if image_b64:
            logging.info(
                f"[{self.name}] Successfully generated image for {recipe_title}."
            )
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"final_recipe_image": image_b64}
                ),
            )
        else:
            logging.warning(
                f"[{self.name}] Could not generate image for {recipe_title}. Replacing with text."
            )
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"final_recipe_image": None}
                ),
            )