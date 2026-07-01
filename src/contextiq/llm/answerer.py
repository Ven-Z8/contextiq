"""Grounded answer synthesis."""

from __future__ import annotations

import re

from pydantic import BaseModel

from contextiq.context.models import ContextPacket
from contextiq.core.config import Settings, get_settings
from contextiq.llm.client import (
    AnthropicLLMClient,
    ExtractiveFallbackClient,
    LLMClient,
    LLMResult,
    OpenRouterLLMClient,
)
from contextiq.llm.prompts import load_answer_prompt


class GroundedAnswer(BaseModel):
    """Answer synthesized from a context packet."""

    text: str
    model: str
    mode: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    warnings: list[str]


class GroundedAnswerer:
    """Synthesize answers using only retrieved context."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or self._default_client(self.settings)
        self.prompt = load_answer_prompt()

    def answer(self, packet: ContextPacket) -> GroundedAnswer:
        """Generate a grounded answer for a context packet."""

        user_prompt = self.prompt.render_user(
            question=packet.question,
            context_packet=packet.to_markdown(),
        )
        result = self._generate_safely(
            system_prompt=self.prompt.system,
            user_prompt=user_prompt,
        )
        return GroundedAnswer(
            text=result.text,
            model=result.model,
            mode=result.mode,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            warnings=[*result.warnings, *self._grounding_warnings(packet, result.text)],
        )

    def _default_client(self, settings: Settings) -> LLMClient:
        if settings.llm_provider == "anthropic":
            if settings.anthropic_api_key is None:
                return ExtractiveFallbackClient()
            return AnthropicLLMClient(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.default_model,
            )
        # Default: OpenRouter (minimax-m3).
        if settings.openrouter_api_key is None:
            return ExtractiveFallbackClient()
        return OpenRouterLLMClient(
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_model,
        )

    def _generate_safely(self, *, system_prompt: str, user_prompt: str) -> LLMResult:
        try:
            return self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.settings.answer_max_tokens,
            )
        except Exception as exc:
            fallback = ExtractiveFallbackClient().generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.settings.answer_max_tokens,
            )
            fallback.warnings.append(f"LLM provider failed; returned extractive fallback: {exc}")
            return fallback

    def _grounding_warnings(self, packet: ContextPacket, answer_text: str) -> list[str]:
        if any(source.block.page is not None for source in packet.sources):
            return []
        if re.search(r"\bpage\s+\d+\b", answer_text, flags=re.IGNORECASE):
            return [
                "Answer mentions a numeric page, but retrieved sources only report page unknown."
            ]
        return []
