import re
from google.cloud import translate_v2 as translate

def translate_text(text: str, target_language: str) -> str:
    """Translates text into the target language, preserving image data.

    Args:
        text: The text to translate.
        target_language: The target language code (e.g., "pt-BR" for Brazilian Portuguese).

    Returns:
        The translated text with image data restored.
    """
    # Find all image data occurrences
    image_placeholders = re.findall(r'(!\[.*?\]\(data:image/png;base64,.*?\))', text)

    # Replace image data with placeholders
    placeholder_text = text
    for i, image_data in enumerate(image_placeholders):
        placeholder = f"__IMAGE_{i}__"
        placeholder_text = placeholder_text.replace(image_data, placeholder)

    # Translate the text without image data
    translate_client = translate.Client()
    result = translate_client.translate(placeholder_text, target_language=target_language)
    translated_text = result["translatedText"]

    # Restore image data
    for i, image_data in enumerate(image_placeholders):
        placeholder = f"__IMAGE_{i}__"
        translated_text = translated_text.replace(placeholder, image_data)

    return translated_text
