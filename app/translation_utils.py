import re
from google.cloud import translate_v2 as translate

def translate_text(text: str, target_language: str) -> str:
    """Translates markdown text into the target language, preserving its formatting and newlines.

    Args:
        text: The markdown text to translate.
        target_language: The target language code (e.g., "pt-BR" for Brazilian Portuguese).

    Returns:
        The translated markdown text with its formatting preserved.
    """
    # 1. Protect Markdown syntax and newlines with placeholders
    placeholders = {}
    placeholder_index = 0

    def placeholder_replacer(match):
        nonlocal placeholder_index
        # Use a more unique placeholder to avoid collision with translated content
        placeholder = f"__MARKDOWN_PLACEHOLDER_{placeholder_index}__"
        placeholders[placeholder] = match.group(0)
        placeholder_index += 1
        return placeholder

    # Regex to find common markdown elements AND newlines
    markdown_regex = re.compile(
        r'(!?\[.*?\]\(.*?\))|'  # Images and links
        r'(\*\*.*?\*\*)|'      # Bold
        r'(__.*?__)|'        # Bold
        r'(\*.*?\*)|'          # Italic
        r'(_.*?_)|'          # Italic
        r'(`.*?`)|'          # Inline code
        r'(^#{1,6} .*)'      # Headers
        r'(\n)'              # Newlines
    , re.MULTILINE)

    protected_text = markdown_regex.sub(placeholder_replacer, text)

    # 2. Translate the text
    translate_client = translate.Client()
    result = translate_client.translate(protected_text, target_language=target_language)
    translated_text = result["translatedText"]

    # 3. Restore Markdown syntax and newlines
    # Iterate in reverse to handle nested placeholders correctly
    # CORRECTED LINE: The index for the split is now -3
    for placeholder, original_text in sorted(placeholders.items(), key=lambda item: int(item[0].split('_')[-3]), reverse=True):
        translated_text = translated_text.replace(placeholder, original_text)

    return translated_text