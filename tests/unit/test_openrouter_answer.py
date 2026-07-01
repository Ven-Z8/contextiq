"""Phase 1: OpenRouter/Nemotron answer client + provider selection.

ponytail: deterministic checks only — mocked HTTP, no live API call.
"""

from __future__ import annotations

import httpx

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
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
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
    settings = Settings(llm_provider="openrouter")
    answerer = GroundedAnswerer(settings=settings)
    assert answerer.client.__class__.__name__ == "ExtractiveFallbackClient"


def test_provider_openrouter_with_key_selects_openrouter_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")  # populated via alias, not init kwarg
    settings = Settings(llm_provider="openrouter")
    answerer = GroundedAnswerer(settings=settings)
    assert isinstance(answerer.client, OpenRouterLLMClient)
    assert answerer.client.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
