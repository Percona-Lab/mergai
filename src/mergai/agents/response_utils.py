"""Shared utilities for parsing agent responses.

This module provides common functions for extracting and parsing JSON responses
from AI agent outputs, handling both direct JSON and markdown-wrapped formats.
"""

import json
import re

from .error import AgentError, AgentErrorType

JSON_BLOCK_RE = re.compile(
    r"```json\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def extract_json_block(text: str) -> str | None:
    """Extract content from first ```json ... ``` block.

    Args:
        text: Text that may contain a JSON code block.

    Returns:
        The JSON string content, or None if not found.
    """
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def parse_response_json(result: dict) -> dict:
    """Parse and validate the response field from agent result.

    Extracts JSON from the 'response' field, handling both:
    - Direct JSON string responses
    - JSON wrapped in markdown code blocks

    Args:
        result: Result dict containing 'response' field.

    Returns:
        Result dict with 'response' parsed as JSON dict.

    Raises:
        AgentError: If 'response' field is missing or invalid JSON.
    """
    if "response" not in result:
        raise AgentError(
            AgentErrorType.PARSING_RESULT, "Invalid response: 'response' field missing"
        )

    response = result["response"]

    # Try to extract from markdown code block first
    extracted = extract_json_block(response)
    if extracted is not None:
        response = extracted

    try:
        response_json = json.loads(response)
        result["response"] = response_json
    except json.JSONDecodeError as e:
        raise AgentError(
            AgentErrorType.PARSING_RESULT, f"JSON decode error: {e}"
        ) from e

    return result
