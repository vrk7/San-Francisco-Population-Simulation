"""LLM wrapper with a pluggable provider (Groq hosted, or local Ollama).

Select the provider with the ``LLM_PROVIDER`` env var ("groq" default, or
"ollama"). Groq is fast and free but rate-limited; Ollama runs locally with no
token cap. Both go through one shared retry-with-backoff loop and raise
:class:`LLMError` on persistent failure, so callers never get a silent empty
result.
"""

import os
import time
from collections.abc import Callable

from dotenv import load_dotenv

from sfsim.constants import DEFAULT_PROVIDER, GROQ_MODEL, OLLAMA_MODEL

load_dotenv()

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0


class MissingAPIKeyError(RuntimeError):
    """Raised when a hosted provider's API key is not configured."""


class LLMError(RuntimeError):
    """Raised when the LLM keeps failing after retries, or is misconfigured."""


def get_api_key() -> str:
    """Return the Groq API key or raise a clear error if it is missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com), or set LLM_PROVIDER=ollama "
            "to run locally."
        )
    return key


def _provider() -> str:
    """The selected provider name, lower-cased."""
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def _groq_call(prompt: str, temperature: float) -> str:
    """One Groq chat completion."""
    from groq import Groq

    client = Groq(api_key=get_api_key())
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _ollama_call(prompt: str, temperature: float) -> str:
    """One local Ollama chat completion (model via OLLAMA_MODEL env or default)."""
    import ollama

    model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"] or ""


_PROVIDERS: dict[str, Callable[[str, float], str]] = {
    "groq": _groq_call,
    "ollama": _ollama_call,
}


def complete(prompt: str, *, temperature: float = 0.5) -> str:
    """Send a single prompt to the selected provider and return the response.

    Retries transient failures with exponential backoff; raises
    :class:`LLMError` if the provider is unknown or every attempt fails.
    """
    provider = _provider()
    caller = _PROVIDERS.get(provider)
    if caller is None:
        raise LLMError(
            f"Unknown LLM_PROVIDER {provider!r}. Use one of: {sorted(_PROVIDERS)}."
        )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return caller(prompt, temperature)
        except MissingAPIKeyError:
            raise  # configuration error — retrying won't help
        except Exception as error:  # noqa: BLE001 — surface after retries as LLMError
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise LLMError(
        f"{provider} request failed after {MAX_RETRIES} attempts: {last_error}"
    )
