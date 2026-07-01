#!/usr/bin/env python3
"""FinanceBench answer-accuracy eval for the ContextIQ corpus pipeline.

Pipeline under test (the corpus-rebuild path):
    hybrid_hits (Qdrant dense+BM25, scoped to the doc) -> token-budgeted packet
    -> minimax-m3 grounded answer -> LLM-judge vs the gold FinanceBench answer.

Prereq: ingest the target 10-Ks first so their chunks land in Qdrant +
data/processed/documents/. Only questions whose document is ingested are scored
(same convention as run_recall.py). Then:

    PYTHONPATH=src <venv>/bin/python evals/financebench/run_answers.py --limit 21

Gate: beat naive-RAG's 0.19 on FinanceBench; target >=0.55.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
DOCUMENTS_DIR = Path("data/processed/documents")
TOKEN_BUDGET = 6000

JUDGE_SYSTEM = (
    "You grade whether a model's answer to a financial question matches the gold answer. "
    "Reply with exactly YES or NO on the first line. Say YES if the model states the same key "
    "figure or fact as the gold answer, ignoring formatting, unit words, sign convention, and "
    "rounding to the gold's precision, plus any extra explanation. Say NO if the key value "
    "differs, is missing, or the model declined to answer."
)


def judge_verdict(raw: str) -> bool:
    """Parse the judge's YES/NO. Anything not clearly YES counts as wrong."""
    return raw.strip().upper().startswith("YES")


def build_docname_to_docid() -> dict[str, str]:
    """Map FinanceBench doc_name (AMD_2022_10K) -> ingested ContextIQ doc_id.

    Document files are <doc_id>-<contenthash>.json where doc_id itself ends in a
    short hash; the FinanceBench stem is the prefix before that.
    """
    mapping: dict[str, str] = {}
    if not DOCUMENTS_DIR.exists():
        return mapping
    for path in DOCUMENTS_DIR.glob("*.json"):
        doc_id = path.stem.rsplit("-", 1)[0]
        mapping[doc_id.rsplit("-", 1)[0]] = doc_id
    return mapping


def _self_check() -> None:
    """The judge parser is the one bit of non-LLM logic that must not drift."""
    assert judge_verdict("YES") is True
    assert judge_verdict("yes, matches ($1,577M)") is True
    assert judge_verdict("NO - model said 1420") is False
    assert judge_verdict("NOT_IN_DOCUMENT") is False
    assert judge_verdict("") is False


def main() -> None:
    ap = argparse.ArgumentParser(description="FinanceBench answer-accuracy")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--limit", type=int, default=0, help="max questions (0 = all ingested)")
    ap.add_argument("--k", type=int, default=40, help="hybrid retrieve depth")
    args = ap.parse_args()

    _self_check()

    # Heavy imports after the self-check so --help / the check stay dependency-free.
    from contextiq.context.models import ContextPacket, ContextSource
    from contextiq.core.config import get_settings
    from contextiq.llm.answerer import GroundedAnswerer
    from contextiq.llm.client import OpenRouterLLMClient
    from contextiq.retrieval.store import LocalDocumentStore
    from contextiq.utils.tokens import TokenCounter

    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    name_to_id = build_docname_to_docid()
    cases = [r for r in rows if r["doc_name"] in name_to_id]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No ingested FinanceBench docs found — ingest some 10-Ks first (see docstring).")
        return

    docs = sorted({c["doc_name"] for c in cases})
    print(f"Scoring {len(cases)} questions across {len(docs)} docs: {docs}\n")

    settings = get_settings()
    enc = TokenCounter()
    answerer = GroundedAnswerer(settings=settings)
    judge = OpenRouterLLMClient(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
    )
    store = LocalDocumentStore()
    store.hybrid_hits("warmup", limit=1)  # init the shared vector index before scoped() clones

    correct = answered = abstained = 0
    for i, r in enumerate(cases):
        scoped = store.scoped(name_to_id[r["doc_name"]])
        hits = scoped.hybrid_hits(r["question"], limit=args.k)

        sources: list[ContextSource] = []
        used = 0
        for hit in hits:
            est = len(enc.encode(hit.block.text))
            if used + est > TOKEN_BUDGET:
                continue
            sources.append(
                ContextSource(
                    block=hit.block,
                    estimated_tokens=est,
                    reason=hit.reason,
                    stages=hit.stages,
                    score=hit.score,
                )
            )
            used += est
        packet = ContextPacket(
            question=r["question"],
            sources=sources,
            token_budget=TOKEN_BUDGET,
            used_tokens=used,
            dropped_candidates=max(len(hits) - len(sources), 0),
        )
        ans = answerer.answer(packet)

        if not sources or ans.text.strip() == "NOT_IN_DOCUMENT":
            abstained += 1
            mark = "~"
        else:
            answered += 1
            jr = judge.generate(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=(
                    f"Question: {r['question']}\n"
                    f"Gold answer: {r['answer']}\n"
                    f"Model answer: {ans.text}\n\nYES or NO?"
                ),
                max_tokens=8,
            )
            if judge_verdict(jr.text):
                correct += 1
                mark = "✓"
            else:
                mark = "✗"
        print(f"[{i + 1}/{len(cases)}] {mark} {r['doc_name']}: {r['question'][:60]}")

    n = len(cases)
    print(f"\nAccuracy:              {correct}/{n} = {correct / n:.3f}   (naive-RAG baseline 0.19)")
    print(f"Answered / Abstained:  {answered} / {abstained}")
    if answered:
        print(f"Correct-when-answered: {correct}/{answered} = {correct / answered:.3f}")


if __name__ == "__main__":
    main()
