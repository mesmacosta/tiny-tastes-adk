import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator
from urllib.parse import urlparse

import google.auth
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.auth.transport import requests
from google.cloud import storage
from google import genai
from google.genai import types

from app.string_utils import process_json_from_recipe

OUTPUT_GCS_PREFIX = "gs://tiny-tastes-generated/generated-videos/"


import json
import logging
# Make sure other necessary imports like BaseAgent, InvocationContext, etc., are present

class VideoGenerationExecutor(BaseAgent):
    """
    A non-LLM agent that executes video generation based on a prompt in the session state.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logging.info(f"[{self.name}] Starting video execution process.")
        video_prompt = ctx.session.state.get("video_prompt")

        if not video_prompt:
            logging.warning(f"[{self.name}] No 'video_prompt' found in state. Skipping.")
            yield Event(author=self.name)
            return

        logging.info(f"[{self.name}] Generating video from prompt: '{video_prompt}'")
        video_uri = await generate_video_from_recipe(video_prompt)
        final_recipe_report = ctx.session.state.get("final_recipe_report")
        final_recipe_image = ctx.session.state.get("final_recipe_image")

        if video_uri:
            logging.info(f"[{self.name}] Successfully generated video.")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"final_video": video_uri,
                                                  "final_recipe_report": final_recipe_report,
                                                  "final_recipe_image" : final_recipe_image}),
            )
        else:
            logging.warning(f"[{self.name}] Video generation failed.")
            yield Event(author=self.name, actions=EventActions(state_delta={"final_video": video_uri,
                                                  "final_recipe_report": final_recipe_report,
                                                  "final_recipe_image" : final_recipe_image}))


# --- NEW, SIMPLER HELPER FUNCTION ---
def create_signed_url_for_gcs_object(
        gcs_uri: str
) -> str | None:
    """
    Generates a time-limited signed URL for a given GCS object URI.

    Args:
        gcs_uri: The full GCS URI of the object (e.g., "gs://bucket-name/path/to/video.mp4").
        expiration_minutes: The duration in minutes for which the URL will be valid.

    Returns:
        A signed URL as a string, or None if an error occurred.
    """
    try:

        credentials, project_id = google.auth.default()

        # Perform a refresh request to get the access token of the current credentials (Else, it's None)
        r = requests.Request()
        credentials.refresh(r)

        # 1. Parse the GCS URI to get the bucket and blob names
        parsed_uri = urlparse(gcs_uri)
        bucket_name = parsed_uri.netloc
        blob_name = parsed_uri.path.lstrip("/")

        # 2. Set up GCS client and get the blob
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        expires = datetime.now() + timedelta(seconds=86400)

        # In case of user credential use, define manually the service account to use (for development purpose only)
        service_account_email = "tiny-tastes-gcs-signer@gcp-tutorials-main.iam.gserviceaccount.com"
        # If you use a service account credential, you can use the embedded email
        if hasattr(credentials, "service_account_email"):
            service_account_email = credentials.service_account_email

        signed_url = blob.generate_signed_url(expiration=expires, service_account_email=service_account_email,
                                       access_token=credentials.token)
        logging.info(f"Generated signed URL for {gcs_uri}")
        return signed_url

    except Exception as e:
        logging.error(f"Failed to create signed URL for {gcs_uri}: {e}")
        return None


# --- AGENT CLASS (No changes needed) ---
# This class remains the same as it reads configuration from the context.


# --- REVISED, SIMPLER VIDEO GENERATION FUNCTION ---
async def generate_video_from_recipe(
        recipe_text: str
) -> str | None:
    """
    Generates a video directly to a GCS location and returns a signed URL.

    Args:
        recipe_text: A string containing the recipe for the video prompt.
        output_gcs_uri_prefix: The GCS path prefix where the video should be saved
                               (e.g., "gs://your-bucket/videos/").

    Returns:
        A signed URL for the generated video if successful, otherwise None.
    """
    if not recipe_text:
        logging.warning("No recipe text provided. Skipping video generation.")
        return None

    # 1. Create the prompt and configure generation settings
    video_prompt = f"A cinematic, high-quality video about the following recipe, give focus on the food, for the recipe name: {recipe_text}"
    generation_config = types.GenerateVideosConfig(
        person_generation="dont_allow",
        aspect_ratio="16:9",
        duration_seconds=8,
        number_of_videos=1,
        output_gcs_uri=OUTPUT_GCS_PREFIX,  # Instruct the API where to save the file
    )

    logging.info(f"Initializing video generation for prompt: '{video_prompt}...'")

    try:
        # 2. Start the asynchronous generation process
        client = genai.Client(vertexai=True, project="gcp-tutorials-main", location="us-central1")
        # --- MODIFICATION: Add the output_gcs_uri parameter ---
        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=video_prompt,
            config=generation_config,
        )
    except Exception as e:
        logging.error(f"An error occurred during the API call: {e}")
        return None

    logging.info("Video generation started. This may take 2-3 minutes...")
    logging.info(f"Operation Name: {operation.name}")

    # 3. Poll for the result (no change here)
    while not operation.done:
        await asyncio.sleep(20)
        logging.info("...Checking operation status...")
        try:
            operation = client.operations.get(operation)
        except Exception as e:
            logging.error(f"Error while polling for status: {e}")
            return None

    # 4. Process the final result
    if operation.done and not operation.error:
        logging.info("Generation complete!")
        response = operation.response

        # The response now contains the GCS URI of the saved video
        for video in response.generated_videos:
            # --- MODIFICATION START ---
            # The URI is now a permanent GCS path, not a temporary file handle.
            final_gcs_uri = video.video.uri
            logging.info(f"Video successfully generated at: {final_gcs_uri}")

            # Simply create a signed URL for the existing GCS object
            signed_url = create_signed_url_for_gcs_object(final_gcs_uri)
            return signed_url
            # --- MODIFICATION END ---
        else:
            logging.warning("Operation completed, but no video was generated.")

    elif operation.error:
        logging.error(f"Video generation failed with an error: {operation.error}")
    else:
        logging.error("Operation did not complete successfully for an unknown reason.")

    return None


# --- REVISED Example Usage ---
async def main():
    """Main function to run the video generation test."""

    test_recipe = """
A cinematic, high-quality video about the following recipe, give focus on the food: A super simple, nutrient-rich, and creamy mash perfect for everyone....
    """

    print("--- Starting Test Video Generation ---")
    signed_video_url = await generate_video_from_recipe(test_recipe)
    print("\n--- Test Complete ---")

    if signed_video_url:
        print(f"\n✅ Successfully generated video directly to GCS!")
        print(f"Signed URL (valid for 60 minutes): {signed_video_url}")
    else:
        print("\n❌ Video generation failed. Please check the logs for errors.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())