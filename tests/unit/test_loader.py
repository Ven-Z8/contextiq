from __future__ import annotations

from types import SimpleNamespace

import pytest
from openpyxl import Workbook
from PIL import Image

from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.loader import DocumentLoader
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.tree_store import TreeStore


def test_loader_extracts_page_from_docling_provenance() -> None:
    item = SimpleNamespace(prov=[SimpleNamespace(page_no=12)])

    assert DoclingStandardExtractor()._page_from_item(item) == 12


def test_loader_maps_docling_labels_to_block_types() -> None:
    assert DoclingStandardExtractor()._block_type_from_label("section_header") == BlockType.HEADING
    assert DoclingStandardExtractor()._block_type_from_label("table") == BlockType.TABLE
    assert DoclingStandardExtractor()._block_type_from_label("picture") == BlockType.FIGURE
    assert DoclingStandardExtractor()._block_type_from_label("text") == BlockType.TEXT


def test_loader_records_docling_fallback_error(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("# Heading\n\nBody text.", encoding="utf-8")
    loader = DocumentLoader()

    def fail_docling(_path, *, page_range=None):
        raise RuntimeError("docling unavailable")

    loader.extractor.extract = fail_docling  # type: ignore[method-assign]

    blocks = loader.load(path)

    assert blocks
    assert blocks[0].metadata["parser"] == "plain_text"
    assert blocks[0].metadata["parser_error"] == "docling unavailable"


def test_loader_uses_native_markdown_parser_for_markdown_files(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Heading\n\n- one\n- two\n\nBody text.", encoding="utf-8")
    loader = DocumentLoader()

    def fail_docling(_path, *, page_range=None):
        raise AssertionError("markdown should not use docling")

    loader.extractor.extract = fail_docling  # type: ignore[method-assign]

    blocks = loader.load(path)

    assert blocks[0].block_type == BlockType.HEADING
    assert blocks[1].metadata["parser"] == "plain_text"
    assert "- one\n- two" in blocks[1].text


def test_loader_strict_docling_raises_parser_errors(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_text("# Heading", encoding="utf-8")
    loader = DocumentLoader(strict_docling=True)

    def fail_docling(_path, *, page_range=None):
        raise RuntimeError("docling unavailable")

    loader.extractor.extract = fail_docling  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="docling unavailable"):
        loader.load(path)


def test_loader_preserves_figure_caption_as_retrievable_text(tmp_path) -> None:
    path = tmp_path / "visual.pdf"
    loader = DocumentLoader()

    item = SimpleNamespace(
        label="picture",
        caption_text="Revenue by segment chart",
        self_ref="#/figures/0",
        prov=[],
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=path)[0]

    assert block.block_type == BlockType.FIGURE
    assert "Revenue by segment chart" in block.text
    assert block.metadata["visual_kind"] == "figure"
    assert block.metadata["caption"] == "Revenue by segment chart"


def test_loader_calls_docling_figure_caption_method(tmp_path) -> None:
    path = tmp_path / "visual.pdf"
    loader = DocumentLoader()

    item = SimpleNamespace(
        label="picture",
        caption_text=lambda: "Architecture overview diagram",
        self_ref="#/figures/0",
        prov=[],
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=path)[0]

    assert block.text == "Figure: Architecture overview diagram"
    assert block.metadata["caption"] == "Architecture overview diagram"


def test_loader_passes_document_to_docling_caption_method(tmp_path) -> None:
    path = tmp_path / "visual.pdf"
    loader = DocumentLoader()

    def caption_text(document):
        assert document is docling_document
        return "Interoperable cloud architecture"

    item = SimpleNamespace(
        label="picture",
        caption_text=caption_text,
        self_ref="#/figures/0",
        prov=[],
    )
    docling_document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=docling_document, path=path)[0]

    assert block.text == "Figure: Interoperable cloud architecture"
    assert block.metadata["caption"] == "Interoperable cloud architecture"


def test_loader_records_docling_figure_bbox_and_saves_image_artifact(tmp_path) -> None:
    source = tmp_path / "visual.pdf"
    loader = DocumentLoader(visuals_dir=tmp_path / "visuals")
    image = Image.new("RGB", (8, 8), color="white")

    item = SimpleNamespace(
        label="picture",
        caption_text=lambda _document: "Architecture diagram",
        self_ref="#/pictures/0",
        prov=[
            SimpleNamespace(
                page_no=7,
                bbox=SimpleNamespace(
                    l=1.0,
                    t=2.0,
                    r=3.0,
                    b=4.0,
                    coord_origin="TOPLEFT",
                ),
            )
        ],
        get_image=lambda _document: image,
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=source)[0]

    assert block.metadata["bbox_l"] == 1.0
    assert block.metadata["bbox_t"] == 2.0
    assert block.metadata["bbox_r"] == 3.0
    assert block.metadata["bbox_b"] == 4.0
    assert block.metadata["image_path"]
    assert (tmp_path / block.metadata["image_path"]).exists()


def test_loader_records_docling_picture_description_and_indexes_it(tmp_path) -> None:
    source = tmp_path / "visual.pdf"
    loader = DocumentLoader(visuals_dir=tmp_path / "visuals")
    image = Image.new("RGB", (8, 8), color="white")

    item = SimpleNamespace(
        label="picture",
        caption_text=lambda _document: "Products and services chart",
        self_ref="#/pictures/0",
        prov=[],
        get_image=lambda _document: image,
        meta=SimpleNamespace(
            description=SimpleNamespace(
                text="Revenue chart shows services growing faster than products.",
                created_by="docling-vlm",
            ),
            classification=None,
            tabular_chart=None,
        ),
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=source)[0]

    assert block.metadata["visual_description"] == (
        "Revenue chart shows services growing faster than products."
    )
    assert block.metadata["visual_description_provider"] == "docling-vlm"
    assert "Visual description: Revenue chart shows services growing faster" in block.text


def test_loader_removes_docling_vlm_stop_tokens_from_picture_description(tmp_path) -> None:
    source = tmp_path / "visual.pdf"
    loader = DocumentLoader(visuals_dir=tmp_path / "visuals")

    item = SimpleNamespace(
        label="picture",
        caption_text=None,
        self_ref="#/pictures/0",
        prov=[],
        get_image=lambda _document: None,
        meta=SimpleNamespace(
            description=SimpleNamespace(
                text="A flow chart showing data services.<end_of_utterance>",
                created_by="docling-vlm",
            ),
            classification=None,
            tabular_chart=None,
        ),
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=source)[0]

    assert block.metadata["visual_description"] == "A flow chart showing data services."
    assert "<end_of" not in block.text


def test_loader_records_docling_picture_classification_metadata(tmp_path) -> None:
    source = tmp_path / "visual.pdf"
    loader = DocumentLoader(visuals_dir=tmp_path / "visuals")
    image = Image.new("RGB", (8, 8), color="white")
    predictions = [
        SimpleNamespace(class_name="bar_chart", confidence=0.91),
        SimpleNamespace(class_name="diagram", confidence=0.22),
    ]

    item = SimpleNamespace(
        label="picture",
        caption_text="Architecture diagram",
        self_ref="#/pictures/0",
        prov=[],
        get_image=lambda _document: image,
        meta=SimpleNamespace(
            description=None,
            classification=SimpleNamespace(
                predictions=predictions,
                get_main_prediction=lambda: predictions[0],
            ),
            tabular_chart=None,
        ),
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=source)[0]

    assert block.metadata["visual_class"] == "bar_chart"
    assert block.metadata["visual_class_confidence"] == 0.91
    assert block.metadata["visual_classes"] == "bar_chart:0.91, diagram:0.22"
    assert "Architecture diagram" in block.text


def test_loader_reads_legacy_docling_picture_annotations(tmp_path) -> None:
    source = tmp_path / "visual.pdf"
    loader = DocumentLoader(visuals_dir=tmp_path / "visuals")

    item = SimpleNamespace(
        label="picture",
        caption_text=None,
        self_ref="#/pictures/0",
        prov=[],
        get_image=lambda _document: None,
        meta=None,
        annotations=[
            SimpleNamespace(
                kind="description",
                text="A diagram showing cloud and data flows.",
                provenance="legacy-vlm",
            ),
            SimpleNamespace(
                kind="classification",
                provenance="classifier",
                predicted_classes=[
                    SimpleNamespace(class_name="flow_chart", confidence=0.83)
                ],
            ),
        ],
    )
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    block = loader.extractor._load_docling_document(document=document, path=source)[0]

    assert block.metadata["visual_description"] == "A diagram showing cloud and data flows."
    assert block.metadata["visual_description_provider"] == "legacy-vlm"
    assert block.metadata["visual_class"] == "flow_chart"
    assert "Visual description: A diagram showing cloud and data flows." in block.text


def test_loader_does_not_plain_text_fallback_binary_pdf(tmp_path) -> None:
    path = tmp_path / "binary.pdf"
    path.write_bytes(b"%PDF-1.4\r\n%\xd3\xf4\xcc\xe1")
    loader = DocumentLoader()

    def fail_docling(_path, *, page_range=None):
        raise RuntimeError("docling parser failed")

    loader.extractor.extract = fail_docling  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="docling parser failed"):
        loader.load(path)


def test_loader_reads_xlsx_sheets_as_table_blocks(tmp_path) -> None:
    path = tmp_path / "nasa-objectives.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Lunar Objectives"
    sheet.append(["Objective ID", "Description", "Theme"])
    sheet.append(["L-001", "Demonstrate lunar surface power", "Power"])
    sheet.append(["L-002", "Validate crew mobility systems", "Mobility"])
    workbook.save(path)

    blocks = DocumentLoader().load(path)

    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.TABLE
    assert blocks[0].section_path == ["Lunar Objectives"]
    assert "Demonstrate lunar surface power" in blocks[0].text
    assert blocks[0].metadata["parser"] == "openpyxl"
    assert blocks[0].metadata["sheet_name"] == "Lunar Objectives"


def test_loader_chunks_large_xlsx_sheets(tmp_path) -> None:
    path = tmp_path / "large-sheet.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Full Decomp"
    sheet.append(["ID", "Description"])
    for index in range(45):
        sheet.append([f"L-{index:03d}", f"Objective row {index}"])
    workbook.save(path)

    blocks = DocumentLoader().load(path)

    assert len(blocks) > 1
    assert blocks[0].metadata["chunk_strategy"] == "table_row_window"
    assert blocks[0].metadata["row_start"] == 1
    assert blocks[0].metadata["parent_block_id"].endswith(":0")


def test_loader_build_tree_persists_a_document_tree(tmp_path) -> None:
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    blocks = [
        DocumentBlock(document_id="d", block_id="d:0", source_path=str(pdf), page=1,
                      block_type=BlockType.HEADING, text="Risk Factors",
                      metadata={"reading_order": 0, "heading_level": 1}),
        DocumentBlock(document_id="d", block_id="d:1", source_path=str(pdf), page=1,
                      block_type=BlockType.TEXT, text="Risks.",
                      metadata={"reading_order": 1}),
    ]
    store = TreeStore(root=tmp_path / "trees")
    loader = DocumentLoader(extractor=StubExtractor(blocks))

    tree = loader.build_tree(pdf, store=store)

    assert tree.nodes[tree.root_id].child_node_ids  # has a section
    assert store.load(tree.document_id) is not None
