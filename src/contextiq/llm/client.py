"""LLM client abstraction."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx
from anthropic import Anthropic
from pydantic import BaseModel, Field

from contextiq.utils.tokens import TokenCounter


class LLMResult(BaseModel):
    """Normalized result from an answer model."""

    text: str
    model: str
    mode: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None = None
    warnings: list[str] = Field(default_factory=list)


# USD per 1M tokens (input, output), keyed by a model-id substring.
_PRICING: dict[str, tuple[float, float]] = {
    "opus": (15.0, 75.0),
    "sonnet": (3.0, 15.0),
    "haiku": (1.0, 5.0),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """Best-effort USD cost from token usage and a model-tier price table.

    Returns None for models not in the table rather than guessing.
    """
    lowered = model.lower()
    for key, (in_price, out_price) in _PRICING.items():
        if key in lowered:
            return round(
                tokens_in / 1_000_000 * in_price + tokens_out / 1_000_000 * out_price, 6
            )
    return None


class LLMClient(ABC):
    """Model provider boundary for answer synthesis."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        """Generate a grounded answer."""


class AnthropicLLMClient(LLMClient):
    """Anthropic-backed LLM client."""

    def __init__(self, *, api_key: str, model: str) -> None:
        # Some host environments (e.g. Claude Desktop) inject ANTHROPIC_AUTH_TOKEN.
        # The SDK turns it into an empty/foreign "Authorization: Bearer" header that
        # breaks x-api-key auth (and passing auth_token="" yields an illegal empty
        # Bearer). Remove it from THIS process's env so x-api-key is used cleanly;
        # contextiq is its own process, so this does not affect the host app.
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        """Generate an answer using Anthropic messages."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "\n".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
        usage = message.usage
        return LLMResult(
            text=text,
            model=self.model,
            mode="anthropic",
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=estimate_cost(self.model, usage.input_tokens, usage.output_tokens),
        )


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLMClient(LLMClient):
    """OpenRouter-backed client over raw httpx.

    ponytail: raw REST, not the OpenAI SDK (CLAUDE.md hard rule). One HTTP call.
    """

    def __init__(
        self, *, api_key: str, model: str, http_client: httpx.Client | None = None
    ) -> None:
        self.api_key = api_key
        self.model = model
        # Injected client in tests; a real one (generous timeout for a slow free tier) otherwise.
        self.http = http_client or httpx.Client(timeout=120.0)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        response = self.http.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        return LLMResult(
            text=text,
            model=self.model,
            mode="openrouter",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            # estimate_cost only knows Claude tiers -> None for OpenRouter models.
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
        )


class ExtractiveFallbackClient(LLMClient):
    """Deterministic fallback used when no API key is configured."""

    def __init__(self) -> None:
        self.encoder = TokenCounter()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        """Return a safe extractive answer instead of calling a remote model."""

        del system_prompt, max_tokens
        source_lines = [
            line
            for line in user_prompt.splitlines()
            if line.startswith("### ") or line.startswith("|")
        ]
        excerpt = "\n".join(source_lines[:18]).strip()
        if excerpt:
            text = (
                "LLM synthesis is disabled because no Anthropic API key is configured. "
                "Here is the strongest retrieved evidence to review:\n\n"
                f"{excerpt}"
            )
        else:
            text = (
                "LLM synthesis is disabled because no Anthropic API key is configured, "
                "and the retrieved context did not include enough extractive evidence."
            )
        return LLMResult(
            text=text,
            model="extractive-fallback",
            mode="extractive_fallback",
            tokens_in=len(self.encoder.encode(user_prompt)),
            tokens_out=len(self.encoder.encode(text)),
            warnings=["Set ANTHROPIC_API_KEY to enable Claude answer synthesis."],
        )
