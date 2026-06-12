from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from contextiq.ingestion.extractors.base import Extractor
from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.loader import DocumentLoader
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


def test_standard_extractor_records_reading_order_and_heading_level(tmp_path) -> None:
    ext = DoclingStandardExtractor()
    heading = SimpleNamespace(label="section_header", level=2, text="Risk Factors",
                              self_ref="#/h", prov=[])
    body = SimpleNamespace(label="text", text="Some prose.", self_ref="#/t", prov=[])
    document = SimpleNamespace(iterate_items=lambda: iter([(heading, 1), (body, 1)]))

    blocks = ext._load_docling_document(document=document, path=tmp_path / "d.pdf")

    assert blocks[0].metadata["reading_order"] == 0
    assert blocks[0].metadata["heading_level"] == 2
    assert blocks[1].metadata["reading_order"] == 1
    assert blocks[1].metadata["layout_label"] == "text"
