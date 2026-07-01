"""Phase 1: OpenRouter answer client + provider selection.

ponytail: deterministic checks only — mocked HTTP, no live API call.
"""

from __future__ import annotations

import time

import httpx
import pytest

from contextiq.core.config import Settings
from contextiq.llm.answerer import GroundedAnswerer
from contextiq.llm.client import OpenRouterLLMClient


def _mock_transport(payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_openrouter_client_parses_response() -> None:
    payload = {
        "choices": [{"message": {"content": "Net sales were $1.2B [b7, page 3]."}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15},
    }
    client = OpenRouterLLMClient(
        api_key="test",
        model="minimax/minimax-m3",
        http_client=httpx.Client(transport=_mock_transport(payload)),
    )
    result = client.generate(system_prompt="s", user_prompt="u", max_tokens=100)
    assert "Net sales" in result.text
    assert "[b7, page 3]" in result.text  # chunk-id citation survives verbatim
    assert result.mode == "openrouter"
    assert result.tokens_in == 120
    assert result.tokens_out == 15


def test_provider_defaults_to_openrouter_and_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # _env_file=None: ignore the developer's real .env so the test is deterministic.
    settings = Settings(llm_provider="openrouter", _env_file=None)
    answerer = GroundedAnswerer(settings=settings)
    assert answerer.client.__class__.__name__ == "ExtractiveFallbackClient"


def test_provider_openrouter_with_key_selects_openrouter_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")  # populated via alias, not init kwarg
    settings = Settings(llm_provider="openrouter", _env_file=None)
    answerer = GroundedAnswerer(settings=settings)
    assert isinstance(answerer.client, OpenRouterLLMClient)
    assert answerer.client.model == "minimax/minimax-m3"


def _counting_transport(responder) -> tuple[httpx.MockTransport, dict]:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return responder(calls["n"])

    return httpx.MockTransport(handler), calls


def test_openrouter_retries_transient_5xx_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # don't actually wait

    def responder(n: int) -> httpx.Response:
        if n < 3:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "recovered"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    transport, calls = _counting_transport(responder)
    client = OpenRouterLLMClient(
        api_key="t", model="minimax/minimax-m3", http_client=httpx.Client(transport=transport)
    )
    result = client.generate(system_prompt="s", user_prompt="u", max_tokens=10)
    assert result.text == "recovered"
    assert calls["n"] == 3  # two transient failures, third succeeds


def test_openrouter_does_not_retry_client_error(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    transport, calls = _counting_transport(lambda n: httpx.Response(400, json={"error": "bad"}))
    client = OpenRouterLLMClient(
        api_key="t", model="minimax/minimax-m3", http_client=httpx.Client(transport=transport)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.generate(system_prompt="s", user_prompt="u", max_tokens=10)
    assert calls["n"] == 1  # 400 is not transient -> no retry
