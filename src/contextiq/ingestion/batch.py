"""Batched ingestion for large PDFs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextiq.ingestion.loader import DocumentLoader
from contextiq.ingestion.models import DocumentBlock
from contextiq.ingestion.pdf_utils import count_pdf_pages, page_batches
from contextiq.ingestion.profiles import QUALITY, IngestProfile, select_ingest_profile

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class BatchIngestResult:
    """Result of a batched or single-pass ingest."""

    blocks: list[DocumentBlock]
    pages_total: int
    pages_done: int
    batches_run: int
    profile_name: str


def renumber_blocks(
    blocks: list[DocumentBlock],
    *,
    document_id: str,
    start_index: int,
) -> tuple[list[DocumentBlock], int]:
    """Assign sequential block ids across merged batch results."""

    renumbered: list[DocumentBlock] = []
    next_index = start_index
    for block in blocks:
        renumbered.append(
            block.model_copy(
                update={
                    "document_id": document_id,
                    "block_id": f"{document_id}:{next_index}",
                }
            )
        )
        next_index += 1
    return renumbered, next_index


class BatchIngestor:
    """Ingest large PDFs in page batches with progress reporting."""

    def __init__(
        self,
        *,
        profile: IngestProfile | None = None,
        strict_docling: bool = False,
    ) -> None:
        self.profile = profile
        self.strict_docling = strict_docling

    def ingest(
        self,
        path: Path,
        *,
        profile: IngestProfile | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BatchIngestResult:
        """Load, chunk, and merge blocks for one document."""

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() not in {".pdf"}:
            loader = DocumentLoader(
                strict_docling=self.strict_docling,
                profile=profile or self.profile or select_ingest_profile(None),
            )
            blocks = loader.load(path)
            return BatchIngestResult(
                blocks=blocks,
                pages_total=0,
                pages_done=0,
                batches_run=1,
                profile_name=(profile or self.profile or QUALITY).name,
            )

        pages_total = count_pdf_pages(path)
        active_profile = profile or self.profile or select_ingest_profile(pages_total)
        if pages_total <= active_profile.batch_page_threshold:
            loader = DocumentLoader(
                strict_docling=self.strict_docling,
                profile=active_profile,
            )
            blocks = loader.load(path)
            if on_progress is not None:
                on_progress(pages_total, pages_total, len(blocks))
            return BatchIngestResult(
                blocks=blocks,
                pages_total=pages_total,
                pages_done=pages_total,
                batches_run=1,
                profile_name=active_profile.name,
            )

        document_id = DocumentLoader(profile=active_profile)._document_id(path)
        merged: list[DocumentBlock] = []
        next_index = 0
        batches = page_batches(pages_total, active_profile.page_batch_size)
        pages_done = 0

        for batch_number, page_range in enumerate(batches, start=1):
            logger.info(
                "Ingest batch %s/%s pages %s-%s",
                batch_number,
                len(batches),
                page_range[0],
                page_range[1],
            )
            loader = DocumentLoader(
                strict_docling=self.strict_docling,
                profile=active_profile,
            )
            batch_blocks = loader.load_pdf_range(path, page_range=page_range)
            renumbered, next_index = renumber_blocks(
                batch_blocks,
                document_id=document_id,
                start_index=next_index,
            )
            merged.extend(renumbered)
            pages_done = page_range[1]
            if on_progress is not None:
                on_progress(pages_done, pages_total, len(merged))

        return BatchIngestResult(
            blocks=merged,
            pages_total=pages_total,
            pages_done=pages_done,
            batches_run=len(batches),
            profile_name=active_profile.name,
        )
