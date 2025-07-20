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

from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import types


def generate_recipe_image(recipe_title: str, recipe_description: str) -> str | None:
    """
    Generates an image for a given recipe using its title and description,
    and returns it as a base64 encoded string.

    Args:
        recipe_title: The title of the recipe (e.g., "Sunshine Sweet Potato Puree").
        recipe_description: A brief summary of the recipe.

    Returns:
        A base64 encoded string of the generated image (PNG format),
        or None if image generation fails.
    """
    prompt = (
        f"Generate a vibrant, photorealistic image of the finished dish for a recipe called '{recipe_title}'. "
        f"The dish is: '{recipe_description}'. "
        "The image should be beautifully styled and appetizing, presented in a small bowl or on a plate suitable for a baby. "
        "The food should be the main focus, set against a clean, bright, and slightly blurred background. "
        "The lighting should be soft and natural, making the dish look delicious and wholesome."
    )

    logging.info(f"Generating image for recipe: {recipe_title}")

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            # The config parameter with response_modalities is not supported
            # in the latest versions of the SDK for this model.
            # The model will return an image by default with a text prompt.
        )

        image_bytes = None
        # Iterate through the response parts to find the image data
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    logging.info(f"Successfully generated image for recipe: {recipe_title}, "
                                 f"mime_type: {part.inline_data.mime_type}, "
                                 f"size: {len(image_bytes)} bytes.")
                    break  # Exit loop once the image is found

        if image_bytes:
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            logging.warning(
                f"No image data was found in the API response for recipe: {recipe_title}. "
                f"Response text: {response.text if hasattr(response, 'text') else 'N/A'}"
            )
            return None

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"A Google API error occurred during image generation for {recipe_title}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during image generation for {recipe_title}: {e}")
        return None

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

    prompt = (
        f"Generate a clear, vibrant, photorealistic image of a single {ingredient_name}, "
        "on a transparent background. The ingredient should be the sole focus, with no shadows. "
        "The final image must be a PNG with a transparent alpha channel, suitable as an icon in a recipe app."
    )

    logging.info(f"Generating image for: {ingredient_name} with prompt: {prompt}")

    try:
        client = genai.Client()
        # Removed the `generation_config` with the unsupported `response_modalities`
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            ),
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
            with open(f"{item.replace(' ', '_')}.png", "wb") as f:
                f.write(base64.b64decode(b64_image))
            # print(f"Saved {item.replace(' ', '_')}.png")
        else:
            print(f"Failed to get image for {item}")