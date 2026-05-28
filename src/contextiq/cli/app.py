"""ContextIQ command-line interface."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from contextiq.context.engine import ContextEngine
from contextiq.evals.retrieval import load_qrels, run_retrieval_eval
from contextiq.ingestion.batch import BatchIngestor
from contextiq.ingestion.profiles import FAST, QUALITY
from contextiq.jobs.models import IngestJobCreate
from contextiq.jobs.runner import run_ingest_job
from contextiq.jobs.store import IngestJobStore
from contextiq.retrieval.store import LocalDocumentStore

app = typer.Typer(help="ContextIQ: context-engineered document intelligence.")
console = Console()


def _resolve_profile_name(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = profile.lower()
    if normalized not in {"fast", "quality"}:
        raise typer.BadParameter("profile must be 'fast' or 'quality'")
    return normalized


@app.command()
def ingest(
    path: Path,
    index: Annotated[
        bool,
        typer.Option(help="Build the local vector index."),
    ] = True,
    profile: Annotated[
        str | None,
        typer.Option(help="Ingest profile: fast or quality."),
    ] = None,
    async_job: Annotated[
        bool,
        typer.Option(
            "--async",
            help="Queue a background ingest job instead of blocking.",
        ),
    ] = False,
) -> None:
    """Parse and store a complex document."""

    profile_name = _resolve_profile_name(profile)
    if async_job:
        job_store = IngestJobStore()
        job = job_store.create(
            IngestJobCreate(
                source_path=str(path.resolve()),
                profile_name=profile_name,
                build_index=index,
            )
        )
        thread = threading.Thread(
            target=run_ingest_job,
            args=(job.job_id,),
            daemon=True,
        )
        thread.start()
        console.print(f"[green]Queued ingest job[/green] {job.job_id}")
        console.print(f"Check status with: contextiq ingest-status {job.job_id}")
        return

    if profile_name == "fast":
        selected_profile = FAST
    elif profile_name == "quality":
        selected_profile = QUALITY
    else:
        selected_profile = None
    ingestor = BatchIngestor(profile=selected_profile)

    def on_progress(pages_done: int, pages_total: int, block_count: int) -> None:
        if pages_total:
            console.print(
                f"[cyan]Progress[/cyan] pages {pages_done}/{pages_total} "
                f"blocks {block_count}"
            )

    result = ingestor.ingest(path, on_progress=on_progress)
    store = LocalDocumentStore()
    store.save_blocks(result.blocks)
    indexed = store.index_blocks(result.blocks) if index else 0
    console.print(
        f"[green]Ingested[/green] {len(result.blocks)} blocks from {path} "
        f"using profile={result.profile_name} batches={result.batches_run}"
    )
    if index:
        console.print(f"[green]Indexed[/green] {indexed} blocks in Qdrant/FastEmbed")


@app.command("ingest-status")
def ingest_status(
    job_id: str,
) -> None:
    """Show background ingest job progress."""

    job = IngestJobStore().get(job_id)
    if job is None:
        raise typer.Exit(code=1)
    console.print(job.model_dump(mode="json"))


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
