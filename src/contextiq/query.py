"""Single seam for answering a question end-to-end (used by CLI and evals)."""

from __future__ import annotations

from contextiq.context.engine import ContextEngine
from contextiq.context.models import ContextPacket
from contextiq.llm.answerer import GroundedAnswer, GroundedAnswerer
from contextiq.retrieval.store import LocalDocumentStore


def answer_question(question: str) -> tuple[GroundedAnswer, ContextPacket]:
    """Build a context packet and synthesize a grounded answer."""
    store = LocalDocumentStore()
    packet = ContextEngine(store=store).build_context(question)
    answer = GroundedAnswerer().answer(packet)
    return answer, packet
