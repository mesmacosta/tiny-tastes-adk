import base64
import logging
import os

#
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.api_core import exceptions as google_exceptions


def generate_ingredient_image(
        ingredient_name: str,
        project_id: str,
        location: str = "us-central1",
) -> str | None:
    """
    Generates an image for a given ingredient name using a Vertex AI Imagen model
    and returns it as a base64 encoded string.

    Args:
        ingredient_name: The name of the ingredient (e.g., "sweet potato").
        project_id: Your Google Cloud project ID.
        location: The Google Cloud location for Vertex AI (e.g., "us-central1").

    Returns:
        A base64 encoded string of the generated image (PNG format),
        or None if image generation fails or no image is returned.
    """
    try:
        # FIX: Initialize the Vertex AI SDK.
        # This uses your project and location and authenticates via Application
        # Default Credentials. Ensure you have run `gcloud auth application-default login`.
        vertexai.init(project=project_id, location=location)
        logging.info(f"Vertex AI initialized for project '{project_id}' in '{location}'.")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}. "
                      "Ensure your project ID is correct and you have authenticated.")
        return None

    # FIX: Use a standard Imagen model designed for high-quality image generation.
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")

    # A more detailed prompt for better, more consistent results.
    prompt = (
        f"a clear, vibrant, photorealistic studio photograph of a single {ingredient_name}, "
        "on a clean, plain white background. The ingredient should be the main focus, centered. "
        "High resolution, professional food photography, no shadows."
    )
    logging.info(f"Generating image for: '{ingredient_name}'")

    try:
        # FIX: Call the correct API method for generating images.
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,  # We only need one image.
        )

        # FIX: The response object contains a list of `GeneratedImage` objects.
        # We need to access the raw bytes from the first image.
        if response.images:
            image_obj = response.images[0]
            # The raw image data is stored in the `_image_bytes` attribute.
            image_bytes = image_obj._image_bytes
            logging.info(f"Successfully generated image for '{ingredient_name}', "
                         f"size: {len(image_bytes)} bytes.")
            # Encode the raw bytes into a base64 string.
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            logging.warning(
                f"No image data found in API response for ingredient: {ingredient_name}. "
                f"Response object: {response}"
            )
            return None

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Google API error for '{ingredient_name}': {e}")
        if "Safety filters" in str(e):
            logging.error("The prompt was likely blocked by safety filters.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred for '{ingredient_name}': {e}")
        return None


if __name__ == '__main__':
    # --- IMPORTANT SETUP ---
    # 1. Have a Google Cloud project with the Vertex AI API enabled.
    #    (https://console.cloud.google.com/vertex-ai)
    # 2. Install the required libraries:
    #    pip install google-cloud-aiplatform Pillow
    # 3. Authenticate your environment in your terminal:
    #    gcloud auth application-default login
    # 4. Set your Google Cloud Project ID below.

    # --- CONFIGURE YOUR PROJECT ID HERE ---
    # It's recommended to use an environment variable for this.
    GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")  # Or hardcode: "your-gcp-project-id"

    # --- SCRIPT EXECUTION ---
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not GCP_PROJECT_ID:
        logging.error("FATAL: Please set the GCP_PROJECT_ID variable in the script or "
                      "as an environment variable.")
    else:
        test_ingredients = ["a whole carrot", "broccoli florets", "a single ripe avocado", "a sprig of rosemary"]
        for item in test_ingredients:
            print("-" * 40)
            b64_image = generate_ingredient_image(item, project_id=GCP_PROJECT_ID)
            if b64_image:
                print(f"SUCCESS: Got base64 image for '{item}'.")
                print(f"         (first 50 chars: {b64_image[:50]}...)")

                # To save and view the image, uncomment the following lines:
                # try:
                #     filename = f"{item.replace(' ', '_')}.png"
                #     with open(filename, "wb") as f:
                #         f.write(base64.b64decode(b64_image))
                #     print(f"         Saved image to '{filename}'")
                # except Exception as e:
                #     print(f"         Could not save file: {e}")

            else:
                print(f"FAILURE: Failed to get image for '{item}'. Check logs for details.")
