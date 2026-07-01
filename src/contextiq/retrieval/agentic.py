"""Simple agentic retrieval: decompose -> retrieve + merge -> rerank.

Targets the measured failure mode (financial tables / line-items not retrieved).
Decomposition rewrites a concept question ("quick ratio") into the underlying
line items ("current assets", "current liabilities", "cash and cash equivalents")
so hybrid search can pull the balance-sheet table. A table-aware LLM rerank then
keeps the most useful blocks (including tables) within the answer's token budget.

No multi-agent orchestration — one decompose call, one rerank call. Built
alongside plain hybrid_hits so it can be measured head-to-head before adoption.
"""

from __future__ import annotations

import json
import logging

from contextiq.llm.client import LLMClient
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.store import LocalDocumentStore

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = (
    "You turn a financial question about a 10-K into focused search queries. "
    "Output ONLY a JSON array of 2-4 short query strings. For ratio or metric "
    "questions, include the underlying financial-statement line items "
    "(e.g. quick ratio -> 'current assets', 'current liabilities', "
    "'cash and cash equivalents', 'accounts receivable'). No prose, only the JSON array."
)

RERANK_SYSTEM = (
    "You rank retrieved passages by how useful they are for answering the question. "
    "Strongly prefer passages that contain the specific figures or tables the answer "
    "needs. Output ONLY a JSON array of the passage numbers (integers) in best-first order."
)


def _extract_json_array(text: str) -> str:
    """minimax may wrap JSON in reasoning/markdown — grab the outermost [...] span."""
    start, end = text.find("["), text.rfind("]")
    return text[start : end + 1] if start != -1 and end > start else "[]"


def decompose(client: LLMClient, question: str) -> list[str]:
    """Return 2-4 sub-queries (line items for ratios/metrics). [] on parse failure."""
    result = client.generate(
        system_prompt=DECOMPOSE_SYSTEM, user_prompt=question, max_tokens=400
    )
    try:
        parsed = json.loads(_extract_json_array(result.text))
        return [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:4]
    except Exception as exc:
        logger.warning("Decompose parse failed; using original query only", exc_info=exc)
        return []


def _rerank(
    client: LLMClient, question: str, candidates: list[RetrievalHit], k: int
) -> list[RetrievalHit]:
    listing = "\n".join(
        f"[{i}] ({c.block.block_type.value}) {c.block.text[:300]}"
        for i, c in enumerate(candidates)
    )
    result = client.generate(
        system_prompt=RERANK_SYSTEM,
        user_prompt=f"Question: {question}\n\nPassages:\n{listing}",
        max_tokens=1000,
    )
    try:
        order = [int(x) for x in json.loads(_extract_json_array(result.text))]
    except Exception:
        order = []
    seen: set[int] = set()
    ranked: list[RetrievalHit] = []
    for idx in order:
        if 0 <= idx < len(candidates) and idx not in seen:
            ranked.append(candidates[idx])
            seen.add(idx)
    # Append any the model didn't rank so nothing is silently dropped before the cut.
    for i, candidate in enumerate(candidates):
        if i not in seen:
            ranked.append(candidate)
    return ranked[:k]


def agentic_retrieve(
    store: LocalDocumentStore,
    question: str,
    client: LLMClient,
    k: int = 15,
    per_query: int = 12,
) -> list[RetrievalHit]:
    """Decompose -> hybrid_hits per sub-query -> merge/dedup -> table-aware rerank."""
    subqueries = [question, *decompose(client, question)]

    pool: dict[str, RetrievalHit] = {}
    for subquery in subqueries:
        for hit in store.hybrid_hits(subquery, limit=per_query):
            existing = pool.get(hit.block.block_id)
            if existing is None or hit.score > existing.score:
                pool[hit.block.block_id] = hit

    candidates = list(pool.values())
    if len(candidates) <= k:
        return candidates
    return _rerank(client, question, candidates, k)
