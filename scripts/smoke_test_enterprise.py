#!/usr/bin/env python3
"""Smoke test: compare legacy vs enterprise search on a set of benchmark queries.

Usage:
    uv run python scripts/smoke_test_enterprise.py
    uv run python scripts/smoke_test_enterprise.py --doc-id apple-2025-10k-e8c0f8fec7
    uv run python scripts/smoke_test_enterprise.py --limit 5

Prints side-by-side results and a score comparison for each query.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextiq.retrieval.store import LocalDocumentStore

# ------------------------------------------------------------------
# Benchmark queries: (expected_profile, query, scope_doc_id or None)
# scope_doc_id scopes the query to a specific document to avoid
# cross-document bleed on company-specific questions.
# ------------------------------------------------------------------
APPLE_DOC = "apple-2025-10k-e8c0f8fec7"
NVIDIA_DOC = "NVIDIA-2025-Annual-Report-218068a832"

BENCHMARK_QUERIES: list[tuple[str, str, str | None]] = [
    # Financial / sparse-heavy
    ("financial_table",   "What was Apple's total net sales in fiscal 2025?",          APPLE_DOC),
    ("financial_table",   "What is the gross margin for the Products segment?",         APPLE_DOC),
    ("financial_table",   "How did Services revenue change compared to prior year?",    APPLE_DOC),
    # Analytical / dense-heavy
    ("analytical",        "Why did operating income increase year over year?",          NVIDIA_DOC),
    ("analytical",        "How does Apple generate revenue from the Services segment?", APPLE_DOC),
    ("analytical",        "What factors drove the increase in R&D spending?",           APPLE_DOC),
    # Risk / prose
    ("risk_section",      "What are Apple's primary regulatory risks?",                 APPLE_DOC),
    ("risk_section",
     "What tariff exposure does Apple face from imported components?",
     APPLE_DOC),
    ("risk_section",      "Describe material litigation Apple is facing",               APPLE_DOC),
    # Narrative
    ("narrative_para",    "What is Apple's strategy for international expansion?",      APPLE_DOC),
    ("narrative_para",    "How does Apple describe its competitive position?",          APPLE_DOC),
]


def _truncate(text: str, max_chars: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _score_result(blocks, query: str) -> float:
    """Relevance heuristic: avg keyword overlap across top-3 results (no bonus terms).

    Measures what fraction of query tokens appear in each returned block,
    averaged across top 3. Penalises returning nothing (score=0).
    No bonuses for metadata — both pipelines judged equally.
    """
    if not blocks:
        return 0.0
    query_tokens = {
        t.lower().strip(".,;:!?\"'()[]")
        for t in query.split()
        if len(t) > 3  # skip stop words like 'the', 'did', 'for', 'how'
    }
    if not query_tokens:
        return 0.0
    scores = []
    for block in blocks[:3]:
        block_tokens = set(block.text.lower().split())
        overlap = len(query_tokens & block_tokens) / len(query_tokens)
        scores.append(overlap)
    return round(sum(scores) / len(scores), 3)


def run_comparison(store: LocalDocumentStore, limit: int, force_doc_id: str | None = None) -> None:
    print(f"\n{'='*80}")
    print(f"  ContextIQ Enterprise Search Smoke Test  (limit={limit})")
    print(f"{'='*80}\n")

    total_legacy = 0.0
    total_enterprise = 0.0
    wins_enterprise = 0
    wins_legacy = 0

    for category, query, scope_doc in BENCHMARK_QUERIES:
        doc_id = force_doc_id or scope_doc
        scope_label = f" [{doc_id.split('-')[0]}]" if doc_id else " [all docs]"
        print(f"[{category.upper()}]{scope_label} {query}")
        print(f"  {'─'*74}")

        scoped = store.scoped(doc_id) if doc_id else store

        # Legacy
        t0 = time.time()
        legacy_results = scoped.search(query, limit=limit)
        legacy_ms = (time.time() - t0) * 1000
        legacy_score = _score_result(legacy_results, query)
        total_legacy += legacy_score

        print(f"  LEGACY  ({legacy_ms:5.0f}ms) kw_score={legacy_score:.3f}")
        for i, b in enumerate(legacy_results[:3]):
            profile = b.metadata.get("content_profile", "—")
            print(f"    [{i+1}] [{profile:18s}] {_truncate(b.text)}")

        # Enterprise
        t0 = time.time()
        enterprise_scored = scoped.search_enterprise_with_scores(query, limit=limit)
        enterprise_ms = (time.time() - t0) * 1000
        enterprise_results = [b for b, _ in enterprise_scored]
        enterprise_score = _score_result(enterprise_results, query)
        total_enterprise += enterprise_score

        verdict = "✓ ENTERPRISE WINS" if enterprise_score > legacy_score else (
                  "✓ LEGACY WINS" if legacy_score > enterprise_score else "  TIE")
        if enterprise_score > legacy_score:
            wins_enterprise += 1
        elif legacy_score > enterprise_score:
            wins_legacy += 1

        print(f"  ENTERP  ({enterprise_ms:5.0f}ms) kw_score={enterprise_score:.3f}  {verdict}")
        for i, (b, colbert_score) in enumerate(enterprise_scored[:3]):
            profile = b.metadata.get("content_profile", "—")
            print(f"    [{i+1}] [{profile:18s}] colbert={colbert_score:.4f}  {_truncate(b.text)}")
        print()

    # Summary
    n = len(BENCHMARK_QUERIES)
    print(f"{'='*80}")
    print(f"  SUMMARY  ({n} queries, limit={limit})")
    print(f"  Legacy avg score:     {total_legacy/n:.3f}")
    print(f"  Enterprise avg score: {total_enterprise/n:.3f}")
    improvement = (total_enterprise - total_legacy) / max(total_legacy, 0.001) * 100
    print(f"  Improvement:          {improvement:+.1f}%")
    print(f"  Enterprise wins: {wins_enterprise}/{n}   Legacy wins: {wins_legacy}/{n}")
    print(f"{'='*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise search smoke test")
    parser.add_argument("--doc-id", help="Scope to a specific document ID")
    parser.add_argument("--limit", type=int, default=8, help="Result limit (default: 8)")
    parser.add_argument(
        "--corpus-path",
        default="data/processed/blocks.json",
        help="Path to blocks.json",
    )
    args = parser.parse_args()

    store = LocalDocumentStore(path=Path(args.corpus_path))

    # Check enterprise collection status
    status = store.enterprise_status()
    if not status.get("available"):
        # Try a probe search anyway — status check can be a false negative
        print(f"⚠️  Status check says: {status.get('reason')} — probing collection anyway...")
        index = store._get_vector_index()
        if index and not index._enterprise_collection_exists():
            print("   Enterprise collection not found. "
                  "Run: uv run python scripts/reindex_enterprise.py first")
            sys.exit(1)
        print("   Collection exists — status check may be a metadata quirk; proceeding.")
    else:
        print(f"✓ Enterprise collection ready: {status}")

    run_comparison(store, limit=args.limit, force_doc_id=args.doc_id or None)


if __name__ == "__main__":
    main()
