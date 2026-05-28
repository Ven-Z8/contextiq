"""PDF helpers for page counting and batch planning."""

from __future__ import annotations

from pathlib import Path


def count_pdf_pages(path: Path) -> int:
    """Return the number of pages in a PDF."""

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def page_batches(total_pages: int, batch_size: int) -> list[tuple[int, int]]:
    """Return inclusive 1-based page ranges for batched Docling conversion."""

    if total_pages <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + batch_size - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges
