"""Run ContextIQ retrieval benchmarks."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from contextiq.evals.retrieval import load_qrels, run_retrieval_eval
from contextiq.retrieval.store import LocalDocumentStore


def main() -> None:
    """Run the same retrieval eval used by the CLI."""

    store = LocalDocumentStore()
    cases = load_qrels(Path("tests/evals/qrels/retrieval_seed.json"))
    report = run_retrieval_eval(
        cases,
        retrieve=lambda question, limit: store.search(question, limit=limit),
        limit=20,
        k=10,
    )
    Console().print(report.to_markdown())


if __name__ == "__main__":
    main()
