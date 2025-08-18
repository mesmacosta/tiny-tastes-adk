import google.generativeai as genai
from app.config import config

def detect_ingredients(image_b64: str) -> str:
    """Detects ingredients from an image using a multimodal LLM.

    Args:
        image_b664 (str): The base64-encoded image.

    Returns:
        str: A comma-separated list of ingredients.
    """
    if not image_b64:
        return ""

    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel('gemini-pro-vision')

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image_b64
        }
    ]

    prompt_parts = [
        "Identify the food ingredients in this image and return them as a comma-separated list.",
        image_parts[0]
    ]

    response = model.generate_content(prompt_parts)
    return response.text
