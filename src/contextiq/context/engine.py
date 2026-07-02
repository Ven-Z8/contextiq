"""Context engineering engine."""

from __future__ import annotations

from contextiq.context.models import ContextPacket, ContextSource
from contextiq.llm.client import LLMClient
from contextiq.retrieval.agentic import agentic_retrieve
from contextiq.retrieval.store import LocalDocumentStore
from contextiq.utils.tokens import TokenCounter


class ContextEngine:
    """Build token-budgeted context packets from hybrid retrieval."""

    def __init__(
        self,
        store: LocalDocumentStore,
        token_budget: int = 6_000,
        agentic_client: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.token_budget = token_budget
        self.encoder = TokenCounter()
        self.agentic_client = agentic_client

    def build_context(self, question: str, limit: int = 40) -> ContextPacket:
        # Agentic (decompose + rerank) when a client is available — validated ~0.79 on
        # FinanceBench's hardest subset vs 0.476 plain; else plain Qdrant hybrid.
        if self.agentic_client is not None:
            hits = agentic_retrieve(self.store, question, self.agentic_client)
        else:
            hits = self.store.hybrid_hits(question, limit=limit)
        sources: list[ContextSource] = []
        used_tokens = 0
        for hit in hits:
            estimated = len(self.encoder.encode(hit.block.text))
            if used_tokens + estimated > self.token_budget:
                continue
            sources.append(
                ContextSource(
                    block=hit.block,
                    estimated_tokens=estimated,
                    reason=hit.reason,
                    stages=hit.stages,
                    score=hit.score,
                )
            )
            used_tokens += estimated
        return ContextPacket(
            question=question,
            sources=sources,
            token_budget=self.token_budget,
            used_tokens=used_tokens,
            dropped_candidates=max(len(hits) - len(sources), 0),
        )
