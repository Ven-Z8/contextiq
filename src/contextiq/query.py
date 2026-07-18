"""Single seam for answering a question end-to-end (used by CLI, evals, and obs)."""

from __future__ import annotations

from contextiq.context.engine import ContextEngine
from contextiq.context.models import ContextPacket
from contextiq.core.config import get_settings
from contextiq.llm.answerer import GroundedAnswer, GroundedAnswerer
from contextiq.llm.client import NvidiaLLMClient, OpenRouterLLMClient
from contextiq.obs import api
from contextiq.retrieval.store import LocalDocumentStore


def _build_packet(question: str) -> ContextPacket:
    settings = get_settings()
    store = LocalDocumentStore()
    client = None
    if settings.agentic:
        if settings.llm_provider == "openrouter" and settings.openrouter_api_key is not None:
            client = OpenRouterLLMClient(
                api_key=settings.openrouter_api_key.get_secret_value(),
                model=settings.openrouter_model,
            )
        elif settings.llm_provider == "nvidia" and settings.nvidia_api_key is not None:
            # Use agentic model for routing/decomposition/reranking
            client = NvidiaLLMClient(
                api_key=settings.nvidia_api_key.get_secret_value(),
                model=settings.nvidia_agentic_model,
                base_url=settings.nvidia_base_url,
            )
    return ContextEngine(store=store, agentic_client=client).build_context(question)


def _synthesize(packet: ContextPacket) -> GroundedAnswer:
    return GroundedAnswerer().answer(packet)


def answer_question(question: str) -> tuple[GroundedAnswer, ContextPacket]:
    """Build a context packet and synthesize a grounded answer (instrumented)."""
    with api.start_run(project="contextiq", label=question) as span:
        with api.observe("contextiq.retrieval"):
            packet = _build_packet(question)
        with api.observe("contextiq.llm.answer"):
            answer = _synthesize(packet)
        api.set_io(span, input=question, output=answer.text)
        api.set_run_metrics(
            span,
            model=answer.model,
            input_tokens=answer.tokens_in,
            output_tokens=answer.tokens_out,
            cost_usd=answer.cost_usd,
            citations_count=len(packet.sources),
            used_tokens=packet.used_tokens,
            dropped_candidates=packet.dropped_candidates,
            token_budget=packet.token_budget,
        )
        return answer, packet
