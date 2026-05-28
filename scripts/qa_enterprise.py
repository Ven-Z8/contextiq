#!/usr/bin/env python3
"""Quick end-to-end Q&A test: enterprise retrieval → Claude answer.

Usage:
    uv run python scripts/qa_enterprise.py
    uv run python scripts/qa_enterprise.py --doc-id apple-2025-10k-e8c0f8fec7
    uv run python scripts/qa_enterprise.py --question "What were Apple's total net sales?"

Wires enterprise retrieval (SPLADE + ColBERT) directly into GroundedAnswerer.
Prints full answer with citations, confidence, and token cost.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextiq.context.models import ContextPacket, ContextSource
from contextiq.ingestion.models import DocumentBlock
from contextiq.llm.answerer import GroundedAnswerer
from contextiq.retrieval.store import LocalDocumentStore
from contextiq.utils.tokens import TokenCounter

# ------------------------------------------------------------------
# Test questions — cover different content profiles and query intents
# ------------------------------------------------------------------
DEFAULT_QUESTIONS = [
    ("apple-2025-10k-e8c0f8fec7",
     "What was Apple's total net sales in fiscal 2025, and how did it break down by Products vs Services?"),

    ("apple-2025-10k-e8c0f8fec7",
     "What factors drove the increase in R&D spending in 2025?"),

    ("apple-2025-10k-e8c0f8fec7",
     "What tariff risks does Apple disclose in its 2025 10-K?"),

    ("apple-2025-10k-e8c0f8fec7",
     "Describe the material litigation Apple is currently facing."),

    ("NVIDIA-2025-Annual-Report-218068a832",
     "Why did NVIDIA's operating income increase so dramatically year over year?"),

    ("NVIDIA-2025-Annual-Report-218068a832",
     "What are NVIDIA's primary risks related to export controls and geopolitical tensions?"),
]

DIVIDER = "=" * 80


def build_enterprise_packet(
    store: LocalDocumentStore,
    question: str,
    doc_id: str | None,
    limit: int = 12,
    token_budget: int = 6_000,
) -> ContextPacket:
    """Retrieve via enterprise pipeline and pack into a ContextPacket."""
    scoped = store.scoped(doc_id) if doc_id else store
    encoder = TokenCounter()

    scored_blocks = scoped.search_enterprise_with_scores(question, limit=limit)

    if not scored_blocks:
        # Fallback to legacy if enterprise returns nothing
        legacy = scoped.search(question, limit=limit)
        scored_blocks = [(b, 0.0) for b in legacy]

    sources: list[ContextSource] = []
    used_tokens = 0

    for block, colbert_score in scored_blocks:
        tokens = len(encoder.encode(block.text))
        if used_tokens + tokens > token_budget:
            continue
        profile = block.metadata.get("content_profile", "generic")
        sources.append(ContextSource(
            block=block,
            estimated_tokens=tokens,
            reason=f"enterprise/{profile}",
            stages=["splade", "rrf", "colbert"],
            score=colbert_score,
        ))
        used_tokens += tokens

    return ContextPacket(
        question=question,
        sources=sources,
        token_budget=token_budget,
        used_tokens=used_tokens,
        dropped_candidates=max(len(scored_blocks) - len(sources), 0),
    )


def run_qa(
    store: LocalDocumentStore,
    question: str,
    doc_id: str | None,
    answerer: GroundedAnswerer,
    q_num: int,
    total: int,
) -> None:
    print(f"\n{DIVIDER}")
    scope = doc_id.split("-")[0].upper() if doc_id else "ALL DOCS"
    print(f"  Q{q_num}/{total} [{scope}]")
    print(f"  {question}")
    print(DIVIDER)

    # Retrieval
    t0 = time.time()
    packet = build_enterprise_packet(store, question, doc_id)
    retrieval_ms = (time.time() - t0) * 1000

    print(f"\n  Retrieved {len(packet.sources)} blocks  "
          f"({packet.used_tokens} tokens, {retrieval_ms:.0f}ms)")

    if not packet.sources:
        print("  ⚠️  No context retrieved — answer will be unsupported")

    # Show top-3 retrieved blocks
    print("\n  Top retrieved blocks:")
    for i, src in enumerate(packet.sources[:3]):
        profile = src.block.metadata.get("content_profile", "—")
        score_str = f"colbert={src.score:.2f}" if src.score else "score=n/a"
        snippet = src.block.text.replace("\n", " ")[:100]
        print(f"    [{i+1}] [{profile:18s}] {score_str}  {snippet}…")

    # LLM answer
    print("\n  Generating answer...")
    t1 = time.time()
    answer = answerer.answer(packet)
    llm_ms = (time.time() - t1) * 1000

    print(f"\n{DIVIDER}")
    print(answer.text)
    print(f"\n{DIVIDER}")
    cost_str = f"${answer.cost_usd:.4f}" if answer.cost_usd else "cost=n/a"
    print(f"  [{answer.model} | {answer.tokens_in}→{answer.tokens_out} tokens | "
          f"{cost_str} | {llm_ms:.0f}ms LLM]")

    if answer.warnings:
        for w in answer.warnings:
            print(f"  ⚠️  {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Q&A end-to-end test")
    parser.add_argument("--doc-id", help="Scope all questions to one document")
    parser.add_argument("--question", help="Ask a single custom question")
    parser.add_argument("--limit", type=int, default=12, help="Blocks to retrieve (default: 12)")
    parser.add_argument(
        "--corpus-path",
        default="data/processed/blocks.json",
        help="Path to blocks.json",
    )
    args = parser.parse_args()

    store = LocalDocumentStore(path=Path(args.corpus_path))

    status = store.enterprise_status()
    if not status.get("available"):
        print(f"⚠️  Enterprise collection not ready: {status.get('reason')}")
        print("   Run: uv run python scripts/reindex_enterprise.py first")
        sys.exit(1)

    print(f"✓ Enterprise collection: {status['points_count']} blocks indexed")

    answerer = GroundedAnswerer()
    print(f"✓ LLM answerer ready: {answerer.settings.default_model}\n")

    if args.question:
        questions = [(args.doc_id, args.question)]
    else:
        questions = DEFAULT_QUESTIONS
        if args.doc_id:
            questions = [(args.doc_id, q) for _, q in questions]

    for i, (doc_id, question) in enumerate(questions, 1):
        run_qa(store, question, doc_id, answerer, i, len(questions))

    print(f"\n{DIVIDER}")
    print(f"  Done — {len(questions)} questions answered")
    print(DIVIDER)


if __name__ == "__main__":
    main()
