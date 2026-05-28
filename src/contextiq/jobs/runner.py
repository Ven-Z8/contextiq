"""Background ingest job execution."""

from __future__ import annotations

import logging
from pathlib import Path

from contextiq.ingestion.batch import BatchIngestor
from contextiq.ingestion.profiles import FAST, QUALITY, IngestProfile, select_ingest_profile
from contextiq.jobs.models import IngestJob, IngestJobStatus
from contextiq.jobs.store import IngestJobStore
from contextiq.retrieval.store import LocalDocumentStore

logger = logging.getLogger(__name__)


def resolve_profile(name: str | None, pages_total: int | None = None) -> IngestProfile:
    if name in {"fast", "quality"}:
        return FAST if name == "fast" else QUALITY
    return select_ingest_profile(pages_total, requested=name)


def run_ingest_job(
    job_id: str,
    *,
    job_store: IngestJobStore | None = None,
    document_store: LocalDocumentStore | None = None,
) -> IngestJob:
    """Execute one ingest job synchronously."""

    store = job_store or IngestJobStore()
    job = store.get(job_id)
    if job is None:
        raise KeyError(f"Ingest job not found: {job_id}")

    job.status = IngestJobStatus.RUNNING
    store.save(job)

    source_path = Path(job.source_path)
    try:
        profile = resolve_profile(job.profile_name)
        ingestor = BatchIngestor(profile=profile)

        def on_progress(pages_done: int, pages_total: int, block_count: int) -> None:
            job.pages_total = pages_total
            job.pages_done = pages_done
            job.blocks_saved = block_count
            store.save(job)

        result = ingestor.ingest(source_path, profile=profile, on_progress=on_progress)
        document_store = document_store or LocalDocumentStore()
        document_store.save_blocks(result.blocks)
        indexed = (
            document_store.index_blocks(result.blocks) if job.build_index else 0
        )

        job.status = IngestJobStatus.COMPLETED
        job.pages_total = result.pages_total
        job.pages_done = result.pages_done
        job.blocks_saved = len(result.blocks)
        job.blocks_indexed = indexed
        job.profile_name = result.profile_name
        job.document_id = result.blocks[0].document_id if result.blocks else None
        job.error = None
        store.save(job)
        return job
    except Exception as exc:
        logger.exception("Ingest job failed", extra={"job_id": job_id})
        job.status = IngestJobStatus.FAILED
        job.error = str(exc)
        store.save(job)
        return job
