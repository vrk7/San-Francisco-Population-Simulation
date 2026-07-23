"""Thin Groq client wrapper. Reads GROQ_API_KEY from the environment."""

import os

from dotenv import load_dotenv

from sfsim.constants import GROQ_MODEL

load_dotenv()


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


def get_api_key() -> str:
    """Return the Groq API key or raise a clear error if it is missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com)."
        )
    return key


def complete(prompt: str, *, temperature: float = 0.5) -> str:
    """Send a single prompt to Groq and return the response text.

    Implemented in Phase 3; validates configuration eagerly here.
    """
    from groq import Groq

    client = Groq(api_key=get_api_key())
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
