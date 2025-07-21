import re
import json
import logging

def process_json_from_recipe(raw_string: str) -> dict | None:
    # 2. Use regex to find and extract the JSON content.
    # This pattern looks for a ```json block and captures everything inside it
    # re.DOTALL makes the '.' special character match any character, including newlines
    match = re.search(r"```json(.*)```", raw_string, re.DOTALL)

    if match:
        # The first captured group (1) is the content inside the parentheses
        json_string = match.group(1).strip()
    else:
        return

    # 3. Parse the extracted JSON string.
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logging.error(
            f"Failed to parse extracted JSON string. Content: %s",
            json_string
        )
        return