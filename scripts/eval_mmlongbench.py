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
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    limit = None if args.limit_docs < 0 else args.limit_docs
    result = evaluate(limit_docs=limit, blocks_per_query=args.blocks_per_query)
    print("\n=== MMLongBench-Doc page-level retrieval (ContextIQ, live path) ===")
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
