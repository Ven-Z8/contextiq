"""CLI: ContextIQ page-level Recall@k on MMLongBench-Doc (downloads from HF)."""

from __future__ import annotations

import argparse
import json
import logging

from contextiq.evals.mmlongbench.runner import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-docs", type=int, default=3, help="docs to evaluate (None=all 135)")
    p.add_argument("--blocks-per-query", type=int, default=30)
    p.add_argument("--pipeline", choices=["legacy", "enterprise", "simple"], default="legacy")
    p.add_argument("--score-mode", choices=["first_page", "page_sum"], default="first_page")
    p.add_argument("--extractor", choices=["standard", "vlm"], default="standard")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    limit = None if args.limit_docs < 0 else args.limit_docs
    result = evaluate(
        limit_docs=limit,
        blocks_per_query=args.blocks_per_query,
        pipeline=args.pipeline,
        score_mode=args.score_mode,
        extractor=args.extractor,
    )
    print(f"\n=== MMLongBench-Doc page-recall (ContextIQ, {args.pipeline}) ===")
    print(json.dumps(result.summary(), indent=2))
    out = "/tmp/mmlb_records.json"
    with open(out, "w") as f:
        json.dump(result.records, f, indent=2)
    print(f"per-question records -> {out}")


if __name__ == "__main__":
    main()
