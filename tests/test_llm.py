"""Tests for the LLM provider switch and retry (no real API / no Ollama needed)."""

import pytest

import sfsim.llm as llm


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(llm.LLMError, match="Unknown LLM_PROVIDER"):
        llm.complete("hi")


def test_provider_dispatches_to_selected_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setitem(llm._PROVIDERS, "ollama", lambda prompt, temp: f"echo:{prompt}")
    assert llm.complete("ping") == "echo:ping"


def test_retries_then_raises_llm_error(monkeypatch):
    calls = {"n": 0}

    def flaky(prompt, temp):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setitem(llm._PROVIDERS, "groq", flaky)
    monkeypatch.setattr(llm, "BACKOFF_BASE_SECONDS", 0.0)  # no real sleeping in tests
    with pytest.raises(llm.LLMError, match="failed after 3 attempts"):
        llm.complete("hi")
    assert calls["n"] == llm.MAX_RETRIES


def test_missing_api_key_is_not_retried(monkeypatch):
    def needs_key(prompt, temp):
        raise llm.MissingAPIKeyError("no key")

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setitem(llm._PROVIDERS, "groq", needs_key)
    with pytest.raises(llm.MissingAPIKeyError):
        llm.complete("hi")
