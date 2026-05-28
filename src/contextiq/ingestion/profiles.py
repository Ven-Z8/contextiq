"""Ingest profiles for small vs large documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestProfile:
    """Docling and batching settings for document ingestion."""

    name: str
    generate_page_images: bool
    generate_picture_images: bool
    enable_picture_enrichment: bool
    table_mode_fast: bool
    batch_page_threshold: int
    page_batch_size: int


FAST = IngestProfile(
    name="fast",
    generate_page_images=False,
    generate_picture_images=False,
    enable_picture_enrichment=False,
    table_mode_fast=True,
    batch_page_threshold=50,
    page_batch_size=50,
)

QUALITY = IngestProfile(
    name="quality",
    generate_page_images=True,
    generate_picture_images=True,
    enable_picture_enrichment=True,
    table_mode_fast=False,
    batch_page_threshold=10_000,
    page_batch_size=50,
)


def select_ingest_profile(page_count: int | None, *, requested: str | None = None) -> IngestProfile:
    """Pick an ingest profile from page count or an explicit name."""

    if requested == "fast":
        return FAST
    if requested == "quality":
        return QUALITY
    if page_count is not None and page_count > FAST.batch_page_threshold:
        return FAST
    return QUALITY
