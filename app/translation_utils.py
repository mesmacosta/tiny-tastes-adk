import re
from google.cloud import translate_v2 as translate


def translate_text(text: str, target_language: str) -> str:
    """Translates markdown text into the target language, preserving its formatting.

    This version uses a split-translate-join approach which is more robust
    for preserving markdown syntax while translating the content.

    Args:
        text: The markdown text to translate.
        target_language: The target language code (e.g., "pt-BR" for Brazilian Portuguese).

    Returns:
        The translated markdown text with its formatting preserved.
    """
    # 1. Define a regex to find all markdown syntax.
    # The capturing group (...) ensures that the syntax itself is kept in the list when we split.
    markdown_syntax_regex = re.compile(
        r'('
        # Match elements that should be preserved as a whole block (content included)
        r'```[\s\S]*?```|'  # Code blocks
        r'`[^`]*?`|'  # Inline code
        r'!?\[.*?\]\(.*?\)|'  # Links and images (content is not translated as a simplification)

        # Match structural and formatting markers that delimit translatable text
        r'^#{1,6}\s+|'  # Header markers (e.g., "## ")
        r'^\s*[\*\-]\s+|'  # Unordered list markers (e.g., "* ")
        r'^\s*\d+\.\s+|'  # Ordered list markers (e.g., "1. ")
        r'\*\*|__|'  # Bold/strong markers
        r'\*|_|'  # Italic/emphasis markers
        r'\n+'  # Capture all newline sequences (handles paragraph breaks)
        r')',
        re.MULTILINE
    )

    # 2. Split the text into a list of alternating content and syntax.
    parts = markdown_syntax_regex.split(text)

    # 3. Collect only the parts that are plain text (i.e., not syntax).
    # We identify text by checking which parts do NOT match our syntax regex.
    text_to_translate = []
    for part in parts:
        # A part is translatable content if it's not empty and not a syntax element.
        if part and not markdown_syntax_regex.fullmatch(part):
            text_to_translate.append(part)

    # If there's no text to translate, return the original text.
    if not text_to_translate:
        return text

    # 4. Translate all collected text parts in a single API call.
    translate_client = translate.Client()
    # The Google Translate API can accept a list of strings.
    results = translate_client.translate(text_to_translate, target_language=target_language)

    # Create an iterator for the translated text results.
    translated_texts_iter = iter([result['translatedText'] for result in results])

    # 5. Reconstruct the document by rejoining the syntax and translated text.
    reconstructed_parts = []
    for part in parts:
        if part and not markdown_syntax_regex.fullmatch(part):
            # This was a text part; replace it with the next translation.
            reconstructed_parts.append(next(translated_texts_iter))
        else:
            # This was a syntax part; keep it as is.
            reconstructed_parts.append(part)

    return "".join(reconstructed_parts)