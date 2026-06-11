from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.extractors.base import Extractor
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.models import BlockType, DocumentBlock


def test_stub_extractor_satisfies_protocol() -> None:
    extractor: Extractor = StubExtractor(
        [DocumentBlock(
            document_id="d", block_id="d:0", source_path="x.pdf",
            block_type=BlockType.HEADING, text="# Title",
        )]
    )
    blocks = extractor.extract(Path("x.pdf"))
    assert extractor.name == "stub"
    assert blocks[0].text == "# Title"


def test_stub_extractor_records_requested_page_range() -> None:
    stub = StubExtractor([])
    stub.extract(Path("x.pdf"), page_range=(1, 10))
    assert stub.last_page_range == (1, 10)


from contextiq.ingestion.loader import DocumentLoader


def test_loader_uses_injected_extractor_for_pdf(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    sentinel = StubExtractor([DocumentBlock(
        document_id="d", block_id="d:0", source_path=str(pdf),
        block_type=BlockType.TEXT, text="hello from stub",
    )])
    loader = DocumentLoader(extractor=sentinel)

    blocks = loader.load(pdf)

    assert sentinel.last_page_range is None
    assert any("hello from stub" in b.text for b in blocks)
