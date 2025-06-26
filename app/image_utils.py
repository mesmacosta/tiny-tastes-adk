# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import logging
import os

from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import types

# Configure the Gemini client
# Ensure GOOGLE_API_KEY is set in your environment or .env file
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    logging.error(
        "GOOGLE_API_KEY not found in environment. Please set it to use Gemini."
    )
    # Allow client initialization to fail later if key is truly missing,
    # but log the common setup issue.
    pass


def generate_ingredient_image(ingredient_name: str) -> str | None:
    """
    Generates an image for a given ingredient name using Gemini
    and returns it as a base64 encoded string.

    Args:
        ingredient_name: The name of the ingredient (e.g., "sweet potato").

    Returns:
        A base64 encoded string of the generated image (PNG format),
        or None if image generation fails or no image is returned.
    """
    try:
        client = genai.Client() # Initialize client here to pick up config
        model = client.models.get(
            "gemini-2.0-flash-preview-image-generation"
        ) # More explicit model fetching
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client or model: {e}")
        return None

    prompt = (
        f"Generate a clear, vibrant, photorealistic image of a single {ingredient_name}, "
        "on a clean, plain white background. The ingredient should be the main focus. "
        "The image should be suitable as an icon in a recipe app."
    )

    logging.info(f"Generating image for: {ingredient_name} with prompt: {prompt}")

    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=types.GenerationConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
            # It's good practice to add safety settings,
            # though defaults are usually reasonable.
            # safety_settings=[
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
        )

        image_bytes = None
        # Iterate through parts to find the image data
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    # Assuming the model returns PNG by default or common case.
                    # Mime type could be checked with part.inline_data.mime_type
                    image_bytes = part.inline_data.data
                    logging.info(f"Successfully generated image for {ingredient_name}, "
                                 f"mime_type: {part.inline_data.mime_type}, "
                                 f"size: {len(image_bytes)} bytes.")
                    break  # Found the image

        if image_bytes:
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            logging.warning(
                f"No image data found in response for ingredient: {ingredient_name}. "
                f"Response: {response.text if hasattr(response, 'text') else 'N/A'}"
            )
            return None

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Google API error during image generation for {ingredient_name}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during image generation for {ingredient_name}: {e}")
        return None

if __name__ == '__main__':
    # Simple test (ensure GOOGLE_API_KEY is set)
    logging.basicConfig(level=logging.INFO)
    test_ingredients = ["carrot", "broccoli florets", "ripe avocado", "nonexistentingredientxyz"]
    for item in test_ingredients:
        print(f"\nTesting with: {item}")
        b64_image = generate_ingredient_image(item)
        if b64_image:
            print(f"Got base64 image for {item} (first 50 chars): {b64_image[:50]}...")
            # To save and view:
            # with open(f"{item.replace(' ', '_')}.png", "wb") as f:
            #     f.write(base64.b64decode(b64_image))
            # print(f"Saved {item.replace(' ', '_')}.png")
        else:
            print(f"Failed to get image for {item}")
