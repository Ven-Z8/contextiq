# Layout-Aware Ingestion → Document Tree — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a PDF into a swappable-engine, layout-preserving extraction that produces a persisted recursive `DocumentTree` with LLM node summaries, wired into ingestion behind a config flag — without breaking the existing vector path or test suite.

**Architecture:** Pull Docling logic out of `DocumentLoader` into a pluggable `Extractor` protocol (`DoclingStandardExtractor` = current behavior + fallback; `DoclingVLMExtractor` = new default, granite-docling-258M on MLX). A new `TreeBuilder` walks ordered blocks by heading level into a recursive `DocumentTree`; a `NodeSummarizer` adds section-local summaries via the existing `LLMClient`; trees persist as JSON. The existing `HierarchyBuilder`/vector path is untouched — the tree is additive.

**Tech Stack:** Python 3.11+, pydantic v2, Docling (standard + VLM pipelines), fastembed, Anthropic SDK (Haiku), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-11-layout-aware-ingestion-design.md`
**Deferred to Plan 2:** RAPTOR bottom-up fallback (§5a), MMLongBench-Doc eval harness (§2), 200-page async wiring (§7).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/contextiq/ingestion/extractors/__init__.py` | Package exports |
| `src/contextiq/ingestion/extractors/base.py` | `Extractor` Protocol |
| `src/contextiq/ingestion/extractors/stub.py` | `StubExtractor` — deterministic, for swap/tests |
| `src/contextiq/ingestion/extractors/docling_standard.py` | `DoclingStandardExtractor` — current Docling logic, moved |
| `src/contextiq/ingestion/extractors/docling_vlm.py` | `DoclingVLMExtractor` — VLM pipeline + MLX auto-select + fallback |
| `src/contextiq/ingestion/loader.py` | MODIFY — thin orchestrator holding an `Extractor`, delegates |
| `src/contextiq/ingestion/tree.py` | `TreeNode`, `DocumentTree`, `TreeBuilder` (heading path) |
| `src/contextiq/ingestion/tree_store.py` | Persist/load `DocumentTree` JSON |
| `src/contextiq/ingestion/summarizer.py` | `NodeSummarizer` — section-local summaries via `LLMClient` |
| `src/contextiq/core/config.py` | MODIFY — add `summary_model`, `enable_vlm_extraction`, `enable_tree_build` |
| `tests/unit/test_extractor_swap.py` | Protocol swap proof |
| `tests/unit/test_docling_vlm_extractor.py` | MLX select + fallback |
| `tests/unit/test_tree_builder.py` | Golden-tree fixtures |
| `tests/unit/test_tree_store.py` | Round-trip persistence |
| `tests/unit/test_node_summarizer.py` | Summary set / skip-small |
| `tests/unit/test_loader.py` | MODIFY — only if delegation changes a signature |

---

## Task 1: `Extractor` protocol + `StubExtractor`

**Files:**
- Create: `src/contextiq/ingestion/extractors/__init__.py`
- Create: `src/contextiq/ingestion/extractors/base.py`
- Create: `src/contextiq/ingestion/extractors/stub.py`
- Test: `tests/unit/test_extractor_swap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extractor_swap.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_extractor_swap.py -v`
Expected: FAIL — `ModuleNotFoundError: contextiq.ingestion.extractors`

- [ ] **Step 3: Write minimal implementation**

```python
# src/contextiq/ingestion/extractors/__init__.py
"""Pluggable document extraction engines."""
```

```python
# src/contextiq/ingestion/extractors/base.py
"""Extractor protocol — the swappable document-reading boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from contextiq.ingestion.models import DocumentBlock


@runtime_checkable
class Extractor(Protocol):
    """Read a document into ordered, citation-preserving blocks."""

    name: str

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        """Return ordered blocks for the document (optionally a page range)."""
        ...
```

```python
# src/contextiq/ingestion/extractors/stub.py
"""Deterministic in-memory extractor for tests and swap proofs."""

from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.models import DocumentBlock


class StubExtractor:
    """Returns a fixed block list; records the last requested page range."""

    name = "stub"

    def __init__(self, blocks: list[DocumentBlock]) -> None:
        self._blocks = blocks
        self.last_page_range: tuple[int, int] | None = None

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        del path
        self.last_page_range = page_range
        return list(self._blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_extractor_swap.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/contextiq/ingestion/extractors tests/unit/test_extractor_swap.py
git commit -m "feat(contextiq): add Extractor protocol + StubExtractor"
```

---

## Task 2: `DoclingStandardExtractor` — move current logic, keep loader tests green

The current Docling logic lives across `DocumentLoader._load_with_docling`, `_load_docling_document`, and ~20 private helpers. We move the **PDF path** into `DoclingStandardExtractor` and have `DocumentLoader` delegate, so the existing `tests/unit/test_loader.py` keeps calling `loader._load_docling_document(...)` and monkeypatching `loader._load_with_docling` unchanged.

**Files:**
- Create: `src/contextiq/ingestion/extractors/docling_standard.py`
- Modify: `src/contextiq/ingestion/loader.py`
- Test: `tests/unit/test_extractor_swap.py` (extend)

- [ ] **Step 1: Write the failing test (loader accepts and uses an injected Extractor)**

```python
# tests/unit/test_extractor_swap.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_extractor_swap.py::test_loader_uses_injected_extractor_for_pdf -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'extractor'`

- [ ] **Step 3: Implement — create the standard extractor and make the loader delegate**

Create `src/contextiq/ingestion/extractors/docling_standard.py`. Move the Docling **conversion + document-walk** logic out of `loader.py` verbatim. The class owns `extract()` and the private helpers (`_load_docling_document`, `_text_from_item`, `_visual_metadata`, `_bbox_metadata`, `_page_from_item`, `_block_type_from_label`, `_label_value`, `_save_visual_image`, picture-description/classification helpers, `_document_id`). It takes the `IngestProfile` + `visuals_dir` + `enable_picture_enrichment` it needs.

```python
# src/contextiq/ingestion/extractors/docling_standard.py
"""Docling standard (TableFormer/layout) pipeline extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.profiles import QUALITY, IngestProfile

logger = logging.getLogger(__name__)


class DoclingStandardExtractor:
    """Current Docling behavior, behind the Extractor protocol."""

    name = "docling_standard"

    def __init__(
        self,
        *,
        profile: IngestProfile | None = None,
        visuals_dir: Path | None = None,
        enable_picture_enrichment: bool | None = None,
    ) -> None:
        self.profile = profile or QUALITY
        self.visuals_dir = visuals_dir or Path("data/processed/visuals")
        self.enable_picture_enrichment = (
            self.profile.enable_picture_enrichment
            if enable_picture_enrichment is None
            else enable_picture_enrichment
        )

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        return self._load_with_docling(path, page_range=page_range)

    # --- moved verbatim from DocumentLoader (do not change behavior) ---
    # _load_with_docling, _docling_converter, _enable_docling_picture_enrichment,
    # _load_docling_document, _text_from_item, _visual_metadata,
    # _docling_picture_metadata, _text_with_visual_description,
    # _docling_picture_description, _clean_visual_text, _docling_picture_classes,
    # _main_prediction, _classification_metadata, _format_prediction,
    # _prediction_class_name, _prediction_confidence, _docling_annotations,
    # _annotation_kind, _page_from_item, _caption_text, _call_caption_text,
    # _save_visual_image, _bbox_metadata, _float_attr, _block_type_from_label,
    # _label_value, _document_id
```

Then in `loader.py`: add an `extractor` constructor param, default-construct a `DoclingStandardExtractor` from the loader's profile/visuals/enrichment, and make `_load_with_docling` / `_load_docling_document` / `_block_type_from_label` / `_page_from_item` **delegate** to the extractor so existing tests still resolve them on the loader:

```python
# loader.py __init__ additions
from contextiq.ingestion.extractors.base import Extractor
from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor

def __init__(self, ..., extractor: Extractor | None = None) -> None:
    ...
    self.extractor = extractor or DoclingStandardExtractor(
        profile=self.profile,
        visuals_dir=self.visuals_dir,
        enable_picture_enrichment=self.enable_picture_enrichment,
    )

# delegating shims (keep existing test call sites working)
def _load_with_docling(self, path, *, page_range=None):
    return self.extractor.extract(path, page_range=page_range)

def _load_docling_document(self, document, path):
    return self.extractor._load_docling_document(document=document, path=path)

def _block_type_from_label(self, label):
    return self.extractor._block_type_from_label(label)

def _page_from_item(self, item):
    return self.extractor._page_from_item(item)
```

Keep the `load()` flow (chunker, plain-text fallback, workbook/markdown branches) on `DocumentLoader`.

- [ ] **Step 4: Run the full loader + swap suite**

Run: `uv run pytest tests/unit/test_loader.py tests/unit/test_extractor_swap.py -v`
Expected: PASS — all existing loader tests green + the new injection test passes.

- [ ] **Step 5: Run the whole suite to confirm no regressions**

Run: `uv run pytest`
Expected: PASS (same count as before + new tests).

- [ ] **Step 6: Commit**

```bash
git add src/contextiq/ingestion/extractors/docling_standard.py src/contextiq/ingestion/loader.py tests/unit/test_extractor_swap.py
git commit -m "refactor(contextiq): move Docling logic into DoclingStandardExtractor behind protocol"
```

---

## Task 3: `DocumentBlock` layout metadata (`reading_order`, `layout_label`, `heading_level`)

The tree needs heading levels; #2 will want reading order and layout labels. Add them as metadata the standard extractor populates.

**Files:**
- Modify: `src/contextiq/ingestion/extractors/docling_standard.py`
- Test: `tests/unit/test_extractor_swap.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extractor_swap.py  (append)
from types import SimpleNamespace

from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_extractor_swap.py::test_standard_extractor_records_reading_order_and_heading_level -v`
Expected: FAIL — `KeyError: 'reading_order'`

- [ ] **Step 3: Implement — enrich metadata in `_load_docling_document`**

In `_load_docling_document`, in the per-item `metadata` dict, add:

```python
metadata["reading_order"] = block_index
metadata["layout_label"] = label
if block_type == BlockType.HEADING:
    metadata["heading_level"] = max(int(getattr(item, "level", 1) or 1), 1)
```

(`block_index` and `label` already exist in that loop; place these right after the existing `metadata = {...}` assignment.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_extractor_swap.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/contextiq/ingestion/extractors/docling_standard.py tests/unit/test_extractor_swap.py
git commit -m "feat(contextiq): record reading_order, layout_label, heading_level on blocks"
```

---

## Task 4: `DoclingVLMExtractor` — VLM pipeline, MLX auto-select, standard fallback

Unit tests do not download the VLM model. We test the deterministic **model-selection** and **fallback** logic; the real conversion path mirrors the standard extractor's `_load_docling_document` walk.

**Files:**
- Create: `src/contextiq/ingestion/extractors/docling_vlm.py`
- Test: `tests/unit/test_docling_vlm_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_docling_vlm_extractor.py
from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.extractors.docling_vlm import DoclingVLMExtractor
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.models import BlockType, DocumentBlock


def test_vlm_selects_mlx_on_apple_silicon() -> None:
    ext = DoclingVLMExtractor(has_mps=True)
    assert ext.vlm_model_name == "GRANITEDOCLING_MLX"


def test_vlm_selects_transformers_without_mps() -> None:
    ext = DoclingVLMExtractor(has_mps=False)
    assert ext.vlm_model_name == "GRANITEDOCLING_TRANSFORMERS"


def test_vlm_falls_back_to_standard_on_conversion_error(tmp_path) -> None:
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    fallback = StubExtractor([DocumentBlock(
        document_id="d", block_id="d:0", source_path=str(pdf),
        block_type=BlockType.TEXT, text="standard fallback block",
    )])
    ext = DoclingVLMExtractor(has_mps=True, fallback=fallback)

    def boom(_path, *, page_range=None):
        raise RuntimeError("vlm model unavailable")

    ext._convert = boom  # type: ignore[method-assign]

    blocks = ext.extract(pdf)
    assert blocks[0].text == "standard fallback block"
    assert ext.name == "docling_vlm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_docling_vlm_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: ...docling_vlm`

- [ ] **Step 3: Implement**

```python
# src/contextiq/ingestion/extractors/docling_vlm.py
"""Docling VLM pipeline extractor (granite-docling-258M, MLX on Apple Silicon)."""

from __future__ import annotations

import logging
from pathlib import Path

from contextiq.ingestion.extractors.base import Extractor
from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor
from contextiq.ingestion.models import DocumentBlock

logger = logging.getLogger(__name__)


def _detect_mps() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class DoclingVLMExtractor:
    """VLM-based extraction with automatic fallback to the standard pipeline."""

    name = "docling_vlm"

    def __init__(
        self,
        *,
        has_mps: bool | None = None,
        fallback: Extractor | None = None,
    ) -> None:
        self._has_mps = _detect_mps() if has_mps is None else has_mps
        self.vlm_model_name = (
            "GRANITEDOCLING_MLX" if self._has_mps else "GRANITEDOCLING_TRANSFORMERS"
        )
        self._fallback = fallback or DoclingStandardExtractor()
        self._standard_walk = DoclingStandardExtractor()

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        try:
            return self._convert(path, page_range=page_range)
        except Exception as exc:
            logger.warning("VLM extraction failed (%s); falling back to standard", exc)
            return self._fallback.extract(path, page_range=page_range)

    def _convert(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        from docling.datamodel import vlm_model_specs  # noqa: PLC0415
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import VlmPipelineOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
        from docling.pipeline.vlm_pipeline import VlmPipeline  # noqa: PLC0415

        options = VlmPipelineOptions(
            vlm_options=getattr(vlm_model_specs, self.vlm_model_name)
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=options
                )
            }
        )
        convert_kwargs = {} if page_range is None else {"page_range": page_range}
        result = converter.convert(str(path), **convert_kwargs)
        # Reuse the standard document-walk so block shape/metadata is identical.
        return self._standard_walk._load_docling_document(
            document=result.document, path=path
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_docling_vlm_extractor.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/contextiq/ingestion/extractors/docling_vlm.py tests/unit/test_docling_vlm_extractor.py
git commit -m "feat(contextiq): add DoclingVLMExtractor with MLX auto-select and standard fallback"
```

---

## Task 5: `TreeNode` + `DocumentTree` models

**Files:**
- Create: `src/contextiq/ingestion/tree.py`
- Test: `tests/unit/test_tree_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tree_builder.py
from __future__ import annotations

from contextiq.ingestion.tree import DocumentTree, TreeNode


def test_document_tree_round_trips_through_json() -> None:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0,
                    page_start=None, page_end=None, parent_id=None)
    tree = DocumentTree(document_id="d", source_path="d.pdf",
                        root_id="d:n0", nodes={"d:n0": root})

    restored = DocumentTree.model_validate_json(tree.model_dump_json())
    assert restored.root_id == "d:n0"
    assert restored.nodes["d:n0"].level == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tree_builder.py::test_document_tree_round_trips_through_json -v`
Expected: FAIL — `ModuleNotFoundError: ...tree`

- [ ] **Step 3: Implement**

```python
# src/contextiq/ingestion/tree.py
"""Recursive document tree for reasoning-based navigation (sub-project #2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    """One node in the document tree (document root, section, or subsection)."""

    node_id: str
    document_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    summary: str = ""
    parent_id: str | None = None
    child_node_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)


class DocumentTree(BaseModel):
    """A document parsed into a recursive node hierarchy."""

    document_id: str
    source_path: str
    root_id: str
    nodes: dict[str, TreeNode]
    page_count: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tree_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contextiq/ingestion/tree.py tests/unit/test_tree_builder.py
git commit -m "feat(contextiq): add TreeNode and DocumentTree models"
```

---

## Task 6: `TreeBuilder` — heading-based recursive build + golden fixtures

**Files:**
- Modify: `src/contextiq/ingestion/tree.py`
- Test: `tests/unit/test_tree_builder.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_tree_builder.py  (append)
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.tree import TreeBuilder


def _heading(doc, idx, level, text, page):
    return DocumentBlock(
        document_id=doc, block_id=f"{doc}:{idx}", source_path="d.pdf", page=page,
        block_type=BlockType.HEADING, text=text,
        metadata={"reading_order": idx, "heading_level": level},
    )


def _body(doc, idx, text, page):
    return DocumentBlock(
        document_id=doc, block_id=f"{doc}:{idx}", source_path="d.pdf", page=page,
        block_type=BlockType.TEXT, text=text, metadata={"reading_order": idx},
    )


def test_tree_builder_nests_h2_under_h1_and_attaches_blocks() -> None:
    blocks = [
        _heading("d", 0, 1, "Part I", 1),
        _heading("d", 1, 2, "Risk Factors", 2),
        _body("d", 2, "Risks here.", 2),
        _heading("d", 3, 2, "Properties", 5),
        _body("d", 4, "Buildings.", 5),
    ]
    tree = TreeBuilder().build(blocks)

    root = tree.nodes[tree.root_id]
    assert root.level == 0
    part = tree.nodes[root.child_node_ids[0]]
    assert part.title == "Part I"
    assert [tree.nodes[c].title for c in part.child_node_ids] == ["Risk Factors", "Properties"]
    risk = tree.nodes[part.child_node_ids[0]]
    assert risk.block_ids == ["d:2"]
    assert risk.page_start == 2


def test_tree_builder_rolls_page_ranges_up_to_parents() -> None:
    blocks = [
        _heading("d", 0, 1, "Part I", 1),
        _body("d", 1, "a", 1),
        _heading("d", 2, 2, "Sub", 3),
        _body("d", 3, "b", 9),
    ]
    tree = TreeBuilder().build(blocks)
    part = tree.nodes[tree.nodes[tree.root_id].child_node_ids[0]]
    assert part.page_start == 1
    assert part.page_end == 9


def test_tree_builder_headingless_doc_yields_single_root_with_all_blocks() -> None:
    blocks = [_body("d", 0, "x", 1), _body("d", 1, "y", 2)]
    tree = TreeBuilder().build(blocks)
    assert list(tree.nodes) == [tree.root_id]
    assert tree.nodes[tree.root_id].block_ids == ["d:0", "d:1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tree_builder.py -v`
Expected: FAIL — `ImportError: cannot import name 'TreeBuilder'`

- [ ] **Step 3: Implement**

```python
# src/contextiq/ingestion/tree.py  (append)
from contextiq.ingestion.models import BlockType, DocumentBlock

_MAX_DEPTH = 6


class TreeBuilder:
    """Build a recursive DocumentTree from heading-leveled blocks (top-down)."""

    def build(self, blocks: list[DocumentBlock]) -> DocumentTree:
        document_id = blocks[0].document_id if blocks else "document"
        source_path = blocks[0].source_path if blocks else ""
        nodes: dict[str, TreeNode] = {}
        counter = 0

        def new_node(title: str, level: int, parent_id: str | None) -> TreeNode:
            nonlocal counter
            node = TreeNode(
                node_id=f"{document_id}:n{counter}", document_id=document_id,
                title=title, level=level, parent_id=parent_id,
            )
            nodes[node.node_id] = node
            counter += 1
            return node

        root = new_node("", 0, None)
        stack: list[TreeNode] = [root]

        for block in blocks:
            if block.block_type == BlockType.HEADING:
                level = self._heading_level(block)
                level = max(1, min(level, _MAX_DEPTH))
                while len(stack) > level:
                    stack.pop()
                while len(stack) < level:
                    # fill skipped levels with the current top to keep a valid chain
                    stack.append(stack[-1])
                parent = stack[-1]
                node = new_node(self._title(block), level, parent.node_id)
                parent.child_node_ids.append(node.node_id)
                stack = stack[:level] + [node]
            else:
                top = stack[-1]
                top.block_ids.append(block.block_id)
                self._extend_pages(top, block.page)

        self._roll_pages_up(root, nodes)
        return DocumentTree(
            document_id=document_id, source_path=source_path,
            root_id=root.node_id, nodes=nodes,
        )

    def _heading_level(self, block: DocumentBlock) -> int:
        meta = block.metadata.get("heading_level")
        if isinstance(meta, int):
            return meta
        stripped = block.text.lstrip()
        hashes = len(stripped) - len(stripped.lstrip("#"))
        return hashes or 1

    def _title(self, block: DocumentBlock) -> str:
        return block.text.lstrip("#").strip()

    def _extend_pages(self, node: TreeNode, page: int | None) -> None:
        if page is None:
            return
        node.page_start = page if node.page_start is None else min(node.page_start, page)
        node.page_end = page if node.page_end is None else max(node.page_end, page)

    def _roll_pages_up(self, node: TreeNode, nodes: dict[str, TreeNode]) -> None:
        for child_id in node.child_node_ids:
            child = nodes[child_id]
            self._roll_pages_up(child, nodes)
            self._extend_pages(node, child.page_start)
            self._extend_pages(node, child.page_end)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tree_builder.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run full suite + lint**

Run: `uv run pytest && uv run ruff check src/contextiq/ingestion/tree.py`
Expected: PASS, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/contextiq/ingestion/tree.py tests/unit/test_tree_builder.py
git commit -m "feat(contextiq): build recursive DocumentTree from heading levels"
```

---

## Task 7: `NodeSummarizer` — section-local summaries via `LLMClient`

Summarizes each node from its **own block text** (no whole-doc context). Small/empty nodes are skipped. Uses the existing `LLMClient` abstraction; failures leave `summary=""` and never raise.

**Files:**
- Create: `src/contextiq/ingestion/summarizer.py`
- Modify: `src/contextiq/core/config.py`
- Test: `tests/unit/test_node_summarizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_node_summarizer.py
from __future__ import annotations

from contextiq.ingestion.summarizer import NodeSummarizer
from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.llm.client import LLMClient, LLMResult


class _FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system_prompt, user_prompt, max_tokens) -> LLMResult:
        self.calls += 1
        return LLMResult(text="A short section summary.", model="fake",
                         mode="fake", tokens_in=1, tokens_out=1)


def _tree_with_one_section() -> tuple[DocumentTree, dict[str, str]]:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0, parent_id=None,
                    child_node_ids=["d:n1"])
    sec = TreeNode(node_id="d:n1", document_id="d", title="Risk Factors", level=1,
                   parent_id="d:n0", block_ids=["d:2"])
    tree = DocumentTree(document_id="d", source_path="d.pdf", root_id="d:n0",
                        nodes={"d:n0": root, "d:n1": sec})
    block_text = {"d:2": "A long paragraph about regulatory and market risks " * 10}
    return tree, block_text


def test_summarizer_sets_summary_on_content_nodes() -> None:
    tree, block_text = _tree_with_one_section()
    llm = _FakeLLM()
    NodeSummarizer(llm=llm, min_words=20).summarize(tree, block_text)
    assert tree.nodes["d:n1"].summary == "A short section summary."
    assert llm.calls == 1  # root (empty) skipped


def test_summarizer_skips_small_nodes() -> None:
    tree, _ = _tree_with_one_section()
    llm = _FakeLLM()
    NodeSummarizer(llm=llm, min_words=20).summarize(tree, {"d:2": "tiny"})
    assert tree.nodes["d:n1"].summary == ""
    assert llm.calls == 0


def test_summarizer_never_raises_on_llm_failure() -> None:
    tree, block_text = _tree_with_one_section()

    class _Boom(LLMClient):
        def generate(self, **_):
            raise RuntimeError("api down")

    NodeSummarizer(llm=_Boom(), min_words=20).summarize(tree, block_text)
    assert tree.nodes["d:n1"].summary == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_node_summarizer.py -v`
Expected: FAIL — `ModuleNotFoundError: ...summarizer`

- [ ] **Step 3: Implement + config**

Add to `src/contextiq/core/config.py` `Settings`:

```python
    summary_model: str = "claude-haiku-4-5"
    enable_vlm_extraction: bool = False
    enable_tree_build: bool = True
    enable_node_summaries: bool = False
```

```python
# src/contextiq/ingestion/summarizer.py
"""Section-local node summaries for the document tree."""

from __future__ import annotations

import logging

from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You summarize one section of a document so a retrieval agent can decide "
    "whether it is relevant. Output ONE or TWO sentences. No preamble."
)
_MAX_INPUT_CHARS = 6000


class NodeSummarizer:
    """Fill TreeNode.summary from each node's own block text."""

    def __init__(self, *, llm: LLMClient, min_words: int = 40,
                 max_tokens: int = 120) -> None:
        self.llm = llm
        self.min_words = min_words
        self.max_tokens = max_tokens

    def summarize(self, tree: DocumentTree, block_text: dict[str, str]) -> DocumentTree:
        for node in tree.nodes.values():
            text = self._node_text(node, block_text)
            if len(text.split()) < self.min_words:
                continue
            node.summary = self._summarize_one(node.title, text)
        return tree

    def _node_text(self, node: TreeNode, block_text: dict[str, str]) -> str:
        parts = [block_text.get(bid, "") for bid in node.block_ids]
        return "\n".join(p for p in parts if p).strip()

    def _summarize_one(self, title: str, text: str) -> str:
        prompt = f"Section title: {title or '(untitled)'}\n\n{text[:_MAX_INPUT_CHARS]}"
        try:
            return self.llm.generate(
                system_prompt=_SYSTEM, user_prompt=prompt, max_tokens=self.max_tokens
            ).text.strip()
        except Exception as exc:
            logger.warning("Node summary failed for %r: %s", title, exc)
            return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_node_summarizer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/contextiq/ingestion/summarizer.py src/contextiq/core/config.py tests/unit/test_node_summarizer.py
git commit -m "feat(contextiq): add NodeSummarizer for section-local tree summaries"
```

---

## Task 8: `DocumentTree` persistence

**Files:**
- Create: `src/contextiq/ingestion/tree_store.py`
- Test: `tests/unit/test_tree_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tree_store.py
from __future__ import annotations

from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.ingestion.tree_store import TreeStore


def _tree() -> DocumentTree:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0, parent_id=None)
    return DocumentTree(document_id="d", source_path="d.pdf", root_id="d:n0",
                        nodes={"d:n0": root})


def test_tree_store_round_trips(tmp_path) -> None:
    store = TreeStore(root=tmp_path / "trees")
    store.save(_tree())
    loaded = store.load("d")
    assert loaded is not None
    assert loaded.root_id == "d:n0"


def test_tree_store_returns_none_for_missing(tmp_path) -> None:
    assert TreeStore(root=tmp_path / "trees").load("nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tree_store.py -v`
Expected: FAIL — `ModuleNotFoundError: ...tree_store`

- [ ] **Step 3: Implement**

```python
# src/contextiq/ingestion/tree_store.py
"""Filesystem persistence for DocumentTree JSON."""

from __future__ import annotations

import re
from pathlib import Path

from contextiq.ingestion.tree import DocumentTree

_SLUG = re.compile(r"[^a-zA-Z0-9_.-]+")


class TreeStore:
    """Read/write DocumentTree JSON under a trees directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("data/processed/trees")

    def _path(self, document_id: str) -> Path:
        slug = _SLUG.sub("-", document_id).strip("-") or "document"
        return self.root / f"{slug[:120]}.json"

    def save(self, tree: DocumentTree) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(tree.document_id)
        path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, document_id: str) -> DocumentTree | None:
        path = self._path(document_id)
        if not path.exists():
            return None
        return DocumentTree.model_validate_json(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tree_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/contextiq/ingestion/tree_store.py tests/unit/test_tree_store.py
git commit -m "feat(contextiq): persist DocumentTree as JSON via TreeStore"
```

---

## Task 9: Wire tree build into ingestion behind config flags

Add a `build_tree()` method to `DocumentLoader` that runs the extractor (no chunking), builds the tree, optionally summarizes, and persists it. Selecting the VLM extractor is driven by `Settings.enable_vlm_extraction`. This keeps `load()` (the existing vector path) untouched.

**Files:**
- Modify: `src/contextiq/ingestion/loader.py`
- Test: `tests/unit/test_loader.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_loader.py  (append)
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.tree_store import TreeStore


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_loader.py::test_loader_build_tree_persists_a_document_tree -v`
Expected: FAIL — `AttributeError: 'DocumentLoader' object has no attribute 'build_tree'`

- [ ] **Step 3: Implement `build_tree` on `DocumentLoader`**

```python
# loader.py additions
from contextiq.ingestion.tree import DocumentTree, TreeBuilder
from contextiq.ingestion.tree_store import TreeStore

def build_tree(
    self,
    path: Path,
    *,
    store: TreeStore | None = None,
    summarizer: "NodeSummarizer | None" = None,
) -> DocumentTree:
    """Extract a document and build (and persist) its recursive tree."""
    blocks = self.extractor.extract(path)
    tree = TreeBuilder().build(blocks)
    if summarizer is not None:
        block_text = {b.block_id: b.text for b in blocks}
        summarizer.summarize(tree, block_text)
    tree.page_count = max((b.page for b in blocks if b.page is not None), default=None)
    if store is not None:
        store.save(tree)
    return tree
```

(Import `NodeSummarizer` lazily inside the method or under `TYPE_CHECKING` to avoid a hard import cycle.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_loader.py::test_loader_build_tree_persists_a_document_tree -v`
Expected: PASS

- [ ] **Step 5: Full suite + lint**

Run: `uv run pytest && uv run ruff check src/contextiq`
Expected: PASS, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/contextiq/ingestion/loader.py tests/unit/test_loader.py
git commit -m "feat(contextiq): build and persist DocumentTree during ingestion"
```

---

## Final verification

- [ ] **Run the whole suite:** `uv run pytest` — all green.
- [ ] **Lint:** `uv run ruff check src/contextiq tests` — clean.
- [ ] **Manual smoke (optional, needs models + a real PDF):**
  ```bash
  uv run python -c "from pathlib import Path; from contextiq.ingestion.loader import DocumentLoader; \
  t = DocumentLoader().build_tree(Path('data/raw/sample-contract.md')); \
  print(t.model_dump_json(indent=2)[:800])"
  ```
  Expected: a `DocumentTree` JSON with a root and child nodes.

---

## What Plan 2 covers (not in scope here)

- **RAPTOR bottom-up fallback** (`ingestion/tree_raptor.py`) — invoked from `TreeBuilder` when heading coverage is below threshold (spec §5a).
- **MMLongBench-Doc eval harness** — `evals/mmlongbench/` loader + ingestion-quality gates (TEDS, reading-order) + page-level Recall@k baseline; `make eval-mmlb` (spec §2).
- **200-page async path** — wire `DoclingVLMExtractor` into `ingestion/batch.py` + `jobs/` with progress; throughput/OOM check (spec §7).
