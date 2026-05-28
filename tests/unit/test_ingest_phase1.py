from __future__ import annotations

from contextiq.ingestion.pdf_utils import page_batches
from contextiq.ingestion.profiles import FAST, QUALITY, select_ingest_profile


def test_select_ingest_profile_uses_fast_for_large_pdfs() -> None:
    assert select_ingest_profile(120).name == "fast"
    assert select_ingest_profile(50).name == "quality"


def test_select_ingest_profile_honors_explicit_name() -> None:
    assert select_ingest_profile(200, requested="quality") is QUALITY
    assert select_ingest_profile(10, requested="fast") is FAST


def test_page_batches_cover_all_pages() -> None:
    assert page_batches(120, 50) == [(1, 50), (51, 100), (101, 120)]


def test_fast_profile_disables_visual_enrichment() -> None:
    assert FAST.enable_picture_enrichment is False
    assert FAST.generate_page_images is False
    assert FAST.table_mode_fast is True
