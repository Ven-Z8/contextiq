"""LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
            cost_usd=None,
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
