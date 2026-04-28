import json
import re

def extract_json_from_output(raw_output: str) -> dict:
    """Extract and sanitize JSON from LLM raw output."""
    text = str(raw_output).strip()
    
    # Fix double curly braces {{ }} -> { }
    text = text.replace('{{', '{').replace('}}', '}')
    
    # Try to find JSON object pattern in the text
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: return raw output wrapped in a dict
        return {"stars": None, "review": text, "_parse_error": True}
