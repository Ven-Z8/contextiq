"""Page-level Recall@k scoring for MMLongBench-Doc.

The benchmark annotates each question with evidence PAGE numbers. We retrieve
blocks, map each block to its source page, dedupe pages in rank order, and score
Recall@k over the distinct top-k pages. This is the headline metric (k=5).
"""

from __future__ import annotations

import re

_INT = re.compile(r"\d+")


def parse_evidence_pages(raw: str | None) -> set[int]:
    """Parse a bracketed page list like '[5, 7]' into a set of ints.

    MMLongBench-Doc uses 1-indexed page numbers; '[]' / '' mean no evidence
    (unanswerable questions).
    """
    if not raw:
        return set()
    return {int(m.group()) for m in _INT.finditer(raw)}


def top_k_pages(ranked_block_pages: list[int | None], k: int) -> list[int]:
    """Distinct page numbers in rank order, skipping None, capped at k."""
    seen: list[int] = []
    for page in ranked_block_pages:
        if page is None or page in seen:
            continue
        seen.append(page)
        if len(seen) >= k:
            break
    return seen


def page_recall_at_k(
    gold_pages: set[int], ranked_block_pages: list[int | None], k: int
) -> float | None:
    """Recall over distinct top-k retrieved pages.

    Returns None when there is no gold evidence (unanswerable questions are scored
    separately, not as retrieval recall).
    """
    if not gold_pages:
        return None
    predicted = set(top_k_pages(ranked_block_pages, k))
    return len(gold_pages & predicted) / len(gold_pages)
