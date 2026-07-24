"""Thin Groq client wrapper. Reads GROQ_API_KEY from the environment."""

import os
import time

from dotenv import load_dotenv

from sfsim.constants import GROQ_MODEL

load_dotenv()

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


class LLMError(RuntimeError):
    """Raised when the Groq API keeps failing after retries."""


def get_api_key() -> str:
    """Return the Groq API key or raise a clear error if it is missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com)."
        )
    return key


def _build_client():
    """Construct a Groq client (import is lazy so tests never need the SDK)."""
    from groq import Groq

    return Groq(api_key=get_api_key())


def complete(prompt: str, *, temperature: float = 0.5) -> str:
    """Send a single prompt to Groq and return the response text.

    Retries transient failures with exponential backoff; raises
    :class:`LLMError` if every attempt fails, so callers never see a silent
    empty result.
    """
    client = _build_client()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 — surface after retries as LLMError
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise LLMError(f"Groq request failed after {MAX_RETRIES} attempts: {last_error}")
