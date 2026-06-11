# ContextIQ Sub-project #1 — Layout-Aware Ingestion → Document Tree

> **Status:** Design approved 2026-06-11 — ready for implementation plan
> **North star:** Agentic Hierarchical RAG ("Approach A"). This is sub-project #1 of 3.
> **Quality bar:** "Undeniable" — reproducible metrics on a recognized public benchmark (MMLongBench-Doc) vs baselines.

---

## 0. Context: where this sits

ContextIQ is being upgraded from a strong-but-conventional vector-RAG stack (SPLADE + ColBERT + RRF) into **Agentic Hierarchical RAG** for 200+ page mixed-bag documents (clean *and* messy). The three layers, built in dependency order, each as its own spec → plan:

1. **Ingestion (this spec)** — pluggable VLM layout-preserving extraction → a recursive document tree with LLM node-summaries.
2. **Index/Retrieval** — keep the vector hybrid AND add a PageIndex-style reasoning tree the LLM navigates.
3. **Orchestration** — an agent that plans the query, chooses tree-reasoning vs vector search, retrieves iteratively, answers with citations + confidence gate.

Everything downstream depends on the document being parsed into a clean, navigable tree, so ingestion is first. It is also the most visually demoable slice.

---

## 1. Goal & non-goals

### Goal
Turn an arbitrary 200+ page PDF (clean or messy/visual-heavy) into:
- **Layout-preserving blocks** — reading order, tables, multi-column text, and figures read correctly, not garbled.
- **A recursive `DocumentTree`** — document → part → section → subsection nesting, each node carrying a short LLM-generated summary.
- All of it behind a **swappable `Extractor` protocol** so the engine (Docling-VLM today; Nemotron / Reducto / Mistral OCR later) is a one-file change.

### Non-goals (explicitly deferred)
- Reasoning retrieval over the tree → sub-project #2.
- The agent loop, query planning, confidence gate → sub-project #3.
- turbopuffer / cloud vector DB → later scale phase.
- Any change to the existing SPLADE/ColBERT vector path. The vector index keeps working unchanged; the tree is **additive**.

---

## 2. Success criteria — the "undeniable" eval pillar

This sub-project ships measured artifacts, not just code. Benchmark: **MMLongBench-Doc** (135 PDFs, avg 47.5 pages; 1,082 questions — 494 single-page, 365 cross-page, 223 unanswerable; evidence from text/table/chart/image/layout; F1 metric where GPT-4o scores 42.7%).

### 2a. Ingestion-quality gates (measured in THIS sub-project)
| Metric | What it proves | Source |
|---|---|---|
| **Table fidelity (TEDS)** | VLM reads tables structurally, not as mangled text | tables in MMLongBench-Doc pages |
| **Reading-order accuracy** | multi-column / newspaper layout is linearized correctly | sampled pages w/ manual order |
| **Block extraction lift vs Docling-standard** | the VLM upgrade beats the current parser | A/B on same pages |
| **Tree-structure correctness** | recursive outline matches the real document TOC | golden-tree fixtures |

### 2b. Page-level retrieval recall scaffold (ready for #2)
MMLongBench-Doc annotates **evidence page numbers**. We build a harness now that, given a question, scores **Recall@k over pages** of whichever retrieval path is wired. In #1 this runs against the current vector path as a *baseline number*; #2 improves it. This is what makes #2's improvement undeniable — the harness and baseline already exist.

### 2c. Reproducibility requirements
- Pinned dataset loader (HF/GitHub), pinned model versions, documented methodology in `docs/benchmarks.md`.
- `make eval-mmlb` runs the gate end-to-end; CI-able.
- Every reported number sits next to a baseline (Docling-standard, and/or published competitor numbers where comparable).

### Exit criteria
- [ ] A 200+ page MMLongBench-Doc PDF ingests without OOM, with progress.
- [ ] VLM extraction beats Docling-standard on table TEDS + reading-order on the sampled set (numbers recorded).
- [ ] `DocumentTree` golden fixtures pass for ≥3 documents of differing structure.
- [ ] Page-level Recall@k baseline number recorded for the current retrieval path.
- [ ] Swapping `Extractor` implementations proven by a stub-extractor test.
- [ ] Existing test suite still green.

---

## 3. Architecture — the two durable abstractions

```
PDF ─► Extractor (protocol) ─► layout blocks ─┬─► TreeBuilder ─► DocumentTree ─► (#2 reasoning)
        DoclingVLM | Nemotron…                 └─► existing vector index (unchanged)
```

### 3a. `Extractor` protocol — `src/contextiq/ingestion/extractors/base.py`
```python
from typing import Protocol
from pathlib import Path
from contextiq.ingestion.models import DocumentBlock

class Extractor(Protocol):
    name: str
    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]: ...
```

Implementations (each its own file under `ingestion/extractors/`):
- `DoclingStandardExtractor` — the **current** `_load_with_docling` logic, moved verbatim out of `DocumentLoader`. Becomes the automatic fallback.
- `DoclingVLMExtractor` — Docling's **VLM pipeline** mode (e.g. SmolDocling/VLM pipeline options). The new default for PDFs.
- `NemotronExtractor` / others — future, not built now. The protocol guarantees they drop in.

`DocumentLoader` is refactored into a thin orchestrator that holds an `Extractor`, selects it by config/profile, and falls back to `DoclingStandardExtractor` on VLM failure or when the VLM model is unavailable. Non-PDF paths (md/txt/xlsx) keep their current loaders.

### 3b. `DocumentTree` / `TreeNode` — `src/contextiq/ingestion/tree.py` (NEW)
The current `HierarchyBuilder` produces a **flat 2-level grouping**, not a tree. This is genuinely new work, not a duplicate.

```python
from pydantic import BaseModel, Field

class TreeNode(BaseModel):
    node_id: str                       # "{document_id}:n{index}"
    document_id: str
    title: str                         # heading text ("" for synthetic root)
    level: int                         # 0 = document root, 1 = H1, ...
    page_start: int | None
    page_end: int | None
    summary: str = ""                  # LLM-generated, 1–2 sentences (§6)
    parent_id: str | None
    child_node_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)  # leaf content blocks

class DocumentTree(BaseModel):
    document_id: str
    source_path: str
    root_id: str
    nodes: dict[str, TreeNode]         # node_id -> node
    page_count: int | None = None
```

`HierarchyBuilder` and `ParentChunk` stay (vector path still uses them) — no breaking change.

---

## 4. Data model changes

- `DocumentBlock` (unchanged shape) gains two `metadata` keys: `reading_order: int` and `layout_label: str` (e.g. `text|table|figure|caption|header|footer`). bbox is already captured.
- New `TreeNode` / `DocumentTree` models in `ingestion/tree.py`.
- No migration of existing stored docs required; trees are generated fresh on (re)ingest.

---

## 5. Tree builder algorithm — `TreeBuilder.build(blocks) -> DocumentTree`

1. Create a synthetic **root** node (level 0) spanning the whole document.
2. Walk blocks in reading order. Maintain a stack of open nodes keyed by heading level (mirrors the `section_stack` logic already in `loader.py`, but builds *nodes* with parent links instead of a flat breadcrumb).
3. On a HEADING block of level *L*: pop the stack to depth *L-1*, create a new `TreeNode` as child of the current top, push it.
4. Non-heading blocks attach their `block_id` to the current top-of-stack node and extend its `page_start/page_end`.
5. After the walk, roll page ranges upward (a parent's range = min/max of descendants).
6. Edge cases: documents with no headings → all blocks under root (single-node tree, still valid); deeply nested or skipped heading levels → clamp to a sane max depth and never break the stack.

Deterministic and unit-testable against golden fixtures.

---

## 6. Node summaries — the bridge to sub-project #2

Each non-trivial node gets a 1–2 sentence `summary`. These are what the #2 reasoning retriever navigates instead of vectors, so building them now is what makes #1 the real foundation.

- Model: **Claude Haiku** (cheap, fast) with **prompt caching** — cache the document body once per doc, summarize each section against it.
- Skip summaries for tiny nodes (below a token threshold) and pure-table leaf nodes (use heading + column headers).
- Batch where the API allows; summaries are best-effort — a summary failure logs and leaves `summary=""`, never blocks ingestion.
- Cost note: bounded and one-time per document; acceptable under "full modern stack, cost secondary," but gated behind a config flag so eval runs can disable it.

---

## 7. 200+ page handling

Reuse existing async/batch infra (`ingestion/batch.py`, `jobs/`). VLM extraction is slower than Docling-standard, so:
- Page-batched extraction (new `Extractor` instance per batch — the Docling OOM mitigation from `docs/blueprint-300-page-documents.md`).
- Progress via the existing async job API (`POST /ingest/async`, `GET /ingest/{job_id}`).
- Tree is built once after all batches merge (it needs the whole heading sequence).

---

## 8. Persistence & file layout

- `data/processed/trees/{document_id}.json` — serialized `DocumentTree`.
- Blocks persist as today (JSON doc store + vector index).
- New extractor model downloads documented in README (VLM model size + first-run download).

---

## 9. Files to create / modify

| File | Action | Description |
|---|---|---|
| `ingestion/extractors/base.py` | CREATE | `Extractor` protocol |
| `ingestion/extractors/docling_standard.py` | CREATE | current `_load_with_docling` moved here |
| `ingestion/extractors/docling_vlm.py` | CREATE | Docling VLM pipeline extractor (new default) |
| `ingestion/extractors/stub.py` | CREATE | deterministic stub for swap/test proof |
| `ingestion/loader.py` | MODIFY | becomes thin orchestrator holding an `Extractor` + fallback |
| `ingestion/tree.py` | CREATE | `TreeNode`, `DocumentTree`, `TreeBuilder` |
| `ingestion/summarizer.py` | CREATE | Haiku node-summary pass (cached, optional) |
| `ingestion/models.py` | MODIFY | add `reading_order` / `layout_label` metadata helpers |
| `evals/mmlongbench/loader.py` | CREATE | dataset loader (docs + evidence-page labels) |
| `evals/mmlongbench/ingestion_gates.py` | CREATE | TEDS, reading-order, tree-correctness, page Recall@k |
| `Makefile` | MODIFY | `make eval-mmlb` target |
| `docs/benchmarks.md` | MODIFY | MMLongBench-Doc methodology + recorded numbers |
| `tests/unit/test_tree_builder.py` | CREATE | golden-tree fixtures |
| `tests/unit/test_extractor_swap.py` | CREATE | protocol swap via stub |
| `tests/unit/test_loader.py` | MODIFY | adapt to orchestrator refactor |

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Docling-VLM slow / heavy on Mac | page-batched + async; `DoclingStandardExtractor` auto-fallback; config to force standard |
| VLM table fidelity worse than TableFormer on some docs | TEDS gate catches it per-doc; fallback path; engine is swappable |
| MMLongBench-Doc is QA-not-retrieval-labeled | derive page-level Recall@k from annotated evidence pages; full QA F1 lands in #3 |
| Summary LLM cost on long docs | Haiku + prompt cache; skip small/table nodes; flag to disable in eval |
| Refactor breaks existing loader tests | move logic verbatim first, refactor behind green tests (TDD) |

---

## 11. Build sequence (slices for the implementation plan)

1. **Refactor to `Extractor` protocol** — extract current logic into `DoclingStandardExtractor`, prove with stub-swap test, keep all tests green. (No behavior change.)
2. **`DoclingVLMExtractor`** — new default, with fallback. Record block-extraction A/B vs standard.
3. **`TreeBuilder` + models** — golden-tree fixtures.
4. **Node summaries** — Haiku + caching, optional flag.
5. **MMLongBench-Doc eval harness** — loader + ingestion gates + page Recall@k baseline; `make eval-mmlb`; numbers into `docs/benchmarks.md`.
6. **200+ page async path** — wire batched VLM into existing job API; OOM/throughput check.

---

*Next: implementation plan via writing-plans skill.*
