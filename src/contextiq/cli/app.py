"""ContextIQ command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from contextiq.context.engine import ContextEngine
from contextiq.evals.retrieval import load_qrels, run_retrieval_eval
from contextiq.ingestion.loader import DocumentLoader
from contextiq.retrieval.store import LocalDocumentStore

app = typer.Typer(help="ContextIQ: context-engineered document intelligence.")
console = Console()


@app.command()
def ingest(
    path: Path,
    index: Annotated[
        bool,
        typer.Option(help="Build the local vector index."),
    ] = True,
) -> None:
    """Parse and store a complex document."""

    loader = DocumentLoader()
    blocks = loader.load(path)
    store = LocalDocumentStore()
    store.save_blocks(blocks)
    indexed = store.index_blocks(blocks) if index else 0
    console.print(f"[green]Ingested[/green] {len(blocks)} blocks from {path}")
    if index:
        console.print(f"[green]Indexed[/green] {indexed} blocks in Qdrant/FastEmbed")


@app.command()
def ask(question: str) -> None:
    """Retrieve context for a question and show the context packet."""

    store = LocalDocumentStore()
    engine = ContextEngine(store=store)
    packet = engine.build_context(question)
    console.print(packet.to_markdown())


@app.command("inspect-context")
def inspect_context() -> None:
    """Show stored document block counts."""

    store = LocalDocumentStore()
    stats = store.stats()
    console.print(stats)


@app.command("eval-retrieval")
def eval_retrieval(
    qrels: Annotated[
        Path,
        typer.Option(help="Path to retrieval qrels JSON."),
    ] = Path("tests/evals/qrels/retrieval_seed.json"),
    limit: Annotated[
        int,
        typer.Option(help="Retriever source limit per query."),
    ] = 20,
    k: Annotated[
        int,
        typer.Option(help="Metric cutoff for Recall@k, Precision@k, and NDCG@k."),
    ] = 10,
) -> None:
    """Run retrieval evals against qrels and print aggregate metrics."""

    store = LocalDocumentStore()
    cases = load_qrels(qrels)
    report = run_retrieval_eval(
        cases,
        retrieve=lambda question, source_limit: store.search(question, limit=source_limit),
        limit=limit,
        k=k,
    )
    console.print(report.to_markdown())
