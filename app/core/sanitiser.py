"""
sanitiser.py — Input sanitisation for all user-supplied strings.

Two sanitisers:
  sanitise_input()      — chat questions (allows newlines, strips control chars)
  sanitise_auth_input() — email / username fields (strict: printable ASCII only)

Neither sanitiser attempts to detect or block specific prompt-injection phrases.
That is the job of the system prompt and the LLM. These functions only remove
characters that have no legitimate place in their respective contexts.
"""

import unicodedata
import logging
from app.core.exceptions import InputValidationError

logger = logging.getLogger(__name__)

# Unicode categories that are never legitimate in user input
_CONTROL_CATEGORIES = {"Cc", "Cs"}   # control characters, surrogates
_ALLOWED_WHITESPACE  = {"\t", "\n"}  # tabs and newlines are fine in chat messages


def sanitise_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitise a raw chat question string.

    Steps:
      1. Reject non-string input immediately.
      2. Normalise Unicode to NFC (canonical decomposition → composition).
      3. Strip null bytes and non-printable control characters
         (preserves tabs and newlines which are legitimate in multi-line questions).
      4. Strip leading / trailing whitespace.
      5. Enforce a hard maximum length.
      6. Reject empty result.

    Raises InputValidationError on any violation.
    """
    if not isinstance(text, str):
        raise InputValidationError("Question must be a text string.")

    # Normalise Unicode
    text = unicodedata.normalize("NFC", text)

    # Remove control characters (keep \t and \n)
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _CONTROL_CATEGORIES
        or ch in _ALLOWED_WHITESPACE
    )

    cleaned = cleaned.strip()

    if not cleaned:
        raise InputValidationError("Question cannot be empty.")

    if len(cleaned) > max_length:
        raise InputValidationError(
            f"Question is too long (max {max_length:,} characters)."
        )

    return cleaned


def sanitise_auth_input(text: str, max_length: int = 254) -> str:
    """
    Sanitise an authentication field (email or username).

    Stricter than sanitise_input — only allows printable ASCII.
    Strips all whitespace, control characters, and non-ASCII Unicode.

    Raises InputValidationError on any violation.
    """
    if not isinstance(text, str):
        raise InputValidationError("Field must be a text string.")

    # Normalise then strip to ASCII printable range (0x20–0x7E)
    text = unicodedata.normalize("NFC", text).strip()
    cleaned = "".join(ch for ch in text if 0x20 <= ord(ch) <= 0x7E)

    if not cleaned:
        raise InputValidationError("Field cannot be empty.")

    if len(cleaned) > max_length:
        raise InputValidationError(
            f"Field is too long (max {max_length} characters)."
        )

    return cleaned