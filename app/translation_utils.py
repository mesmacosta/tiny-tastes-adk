from google.cloud import translate_v2 as translate

def translate_text(text: str, target_language: str) -> str:
    """Translates text into the target language.

    Args:
        text: The text to translate.
        target_language: The target language code (e.g., "pt-BR" for Brazilian Portuguese).

    Returns:
        The translated text.
    """
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result["translatedText"]
