"""Context engineering engine."""

from __future__ import annotations

from contextiq.context.models import ContextPacket, ContextSource
from contextiq.retrieval.store import LocalDocumentStore
from contextiq.utils.tokens import TokenCounter


class ContextEngine:
    """Build token-budgeted context packets from hybrid retrieval."""

    def __init__(self, store: LocalDocumentStore, token_budget: int = 6_000) -> None:
        self.store = store
        self.token_budget = token_budget
        self.encoder = TokenCounter()

    def build_context(self, question: str, limit: int = 40) -> ContextPacket:
        # Live retrieve: Qdrant hybrid dense+BM25 (validated on FinanceBench, 0.476 vs 0.19).
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
