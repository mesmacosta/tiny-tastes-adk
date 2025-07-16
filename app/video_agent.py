import logging
import time
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.invocation_context import InvocationContext
import google.generativeai as genai
from google.generativeai import types

class VideoGeneratorAgent(BaseAgent):
    """
    An agent that generates a video from a recipe.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logging.info(f"[{self.name}] Starting video generation process.")
        final_recipe_report = ctx.session.state.get("final_recipe_report")

        if not final_recipe_report:
            logging.warning(
                f"[{self.name}] No 'final_recipe_report' found in state. Skipping."
            )
            yield Event(author=self.name)
            return

        video_prompt = f"A video about the following recipe: {final_recipe_report}"

        generation_config = types.GenerateVideosConfig(
            person_generation="dont_allow",
            aspect_ratio="16:9",
            duration_seconds=8,
            number_of_videos=1,
        )

        logging.info(f"Initializing video generation for prompt: '{video_prompt}'")

        try:
            client = genai.Client()
            operation = client.generate_videos(
                model="veo-2.0-generate-001",
                prompt=video_prompt,
                config=generation_config,
            )
        except Exception as e:
            logging.error(f"An error occurred during API call: {e}")
            yield Event(author=self.name)
            return

        logging.info("Video generation started. This may take 2-3 minutes...")
        logging.info(f"Operation Name: {operation.operation.name}")

        while not operation.done:
            logging.info("...Waiting for generation to complete...")
            time.sleep(20)
            try:
                operation = client.operations.get(operation.operation.name)
            except Exception as e:
                logging.error(f"An error occurred while polling for status: {e}")
                break

        if operation.done and not operation.error:
            logging.info("Generation complete!")
            response = operation.result()

            for i, video in enumerate(response.generated_videos):
                video_url = video.video.uri
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={"final_video": video_url}
                    ),
                )
                return

        elif operation.error:
            logging.error(f"Video generation failed with an error: {operation.error.message}")
        else:
            logging.error("Operation did not complete successfully.")

        yield Event(author=self.name)
