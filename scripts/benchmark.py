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
    lexical_report = run_retrieval_eval(
        cases,
        retrieve=lambda question, limit: store._search_lexical(question, limit=limit),
        limit=20,
        k=10,
    )
    contextiq_report = run_retrieval_eval(
        cases,
        retrieve=lambda question, limit: store.search(question, limit=limit),
        limit=20,
        k=10,
    )
    console = Console()
    console.print("# Retrieval Benchmark Comparison")
    console.print("")
    console.print("| Metric | Lexical-only baseline | ContextIQ |")
    console.print("| --- | ---: | ---: |")
    console.print(f"| Queries | {lexical_report.query_count} | {contextiq_report.query_count} |")
    for metric in lexical_report.aggregate:
        console.print(
            f"| {metric} | {lexical_report.aggregate[metric]:.3f} | "
            f"{contextiq_report.aggregate[metric]:.3f} |"
        )


if __name__ == "__main__":
    main()
