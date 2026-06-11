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

This sub-project ships measured artifacts, not just code. Benchmark: **MMLongBench-Doc** (135 PDFs, avg 47.5 pages; **1,082** questions — 494 single-page, 365 cross-page, 223 unanswerable; evidence from text/table/chart/image/layout; **F1 metric, best baseline GPT-4o = 44.9%**). Distribution: HuggingFace `yubo2333/MMLongBench-Doc` (Parquet) + [GitHub](https://github.com/mayubo2333/MMLongBench-Doc). Each question carries an **`evidence_pages`** annotation (verified 2026-06-11) — this is what makes §2b's page-level Recall@k real, not aspirational.

> Verified notes (research 2026-06-11): the 44.9% GPT-4o F1 supersedes an earlier 42.7% figure; some sources cite 1,091 questions vs 1,082 (minor count discrepancy — pin the exact split at implementation from the HF Parquet). Dataset **license is not yet confirmed** — verify before redistributing any derived artifact.

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
- `DoclingVLMExtractor` — Docling's **VLM pipeline**, pinned to the real current API (verified 2026-06-11): `VlmPipeline` + `VlmPipelineOptions` with `vlm_options` from `docling.datamodel.vlm_model_specs`. Default model **granite-docling-258M** (`GRANITEDOCLING_MLX` on Apple Silicon, `GRANITEDOCLING_TRANSFORMERS` elsewhere). CLI equivalent: `docling --pipeline vlm FILE`. New default for PDFs.
  ```python
  from docling.pipeline.vlm_pipeline import VlmPipeline
  from docling.datamodel.pipeline_options import VlmPipelineOptions
  from docling.datamodel import vlm_model_specs
  opts = VlmPipelineOptions(vlm_options=vlm_model_specs.GRANITEDOCLING_MLX)  # MLX on macOS/MPS
  # DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)})
  ```
  **MLX is mandatory on Mac:** measured 6.15s/page (MLX) vs 102.2s/page (Transformers) on M3 Max — a ~16× gap. Auto-select MLX when MPS is available.
- `NemotronExtractor` — **NVIDIA Nemotron-Parse 1.1** (open-weight, layout-aware structured output). Confirmed real and a strong second engine; stubbed now, built when Venkat swaps. The protocol guarantees it drops in. (API option tier: Mistral OCR 3 — SOTA commodity-priced — is the cloud fallback if ever needed.)

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
6. Edge cases: deeply nested or skipped heading levels → clamp to a sane max depth and never break the stack.

### 5a. Headingless / no-TOC documents — RAPTOR fallback (mixed-bag robustness)
The top-down heading walk above is the PageIndex-style approach and is excellent on *structured* docs. But our stated target is **mixed-bag**, where a document may have **no usable heading structure** (scans, brochures, newspapers). A single-root tree there is useless for reasoning retrieval. So when heading coverage is below a threshold, `TreeBuilder` falls back to a **RAPTOR-style bottom-up build** ([RAPTOR, arXiv:2401.18059](https://arxiv.org/abs/2401.18059)): embed leaf blocks → cluster → LLM-summarize each cluster into a synthetic parent node → recurse until one root. This yields a real navigable tree even with zero headings. Heading-based (top-down) is preferred when structure exists; RAPTOR (bottom-up) is the fallback — both produce the same `DocumentTree` shape, so #2 navigates them identically.

Deterministic for the heading path; the RAPTOR path is seeded/clustering-based and tested for *shape and coverage* rather than exact node identity.

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

**Honest throughput target (corrected by research):** at 6.15s/page (granite/SmolDocling MLX, M3 Max), a 200-page doc is **~20 min** of VLM extraction — the old blueprint's "<5 min" target is **not** achievable with a VLM pass and is hereby retired for the VLM path. VLM ingestion is **async-batch only, never interactive**. Mitigations: MLX on Apple Silicon (16× over Transformers), and `DoclingStandardExtractor` remains the fast path for users who opt out of VLM. On a non-MLX Mac Mini, expect materially slower — default such machines to the standard extractor unless VLM is explicitly requested.

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
| `ingestion/tree.py` | CREATE | `TreeNode`, `DocumentTree`, `TreeBuilder` (top-down heading path) |
| `ingestion/tree_raptor.py` | CREATE | RAPTOR bottom-up fallback for headingless docs (§5a) |
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
| Docling-VLM slow / heavy on Mac (~6s/page MLX, ~20 min/200pp) | MLX auto-select (16× faster); async-batch only; `DoclingStandardExtractor` auto-fallback; default non-MLX machines to standard |
| VLM table fidelity worse than TableFormer on some docs | TEDS gate catches it per-doc; fallback path; engine is swappable |
| MMLongBench-Doc is QA-not-retrieval-labeled | derive page-level Recall@k from annotated evidence pages; full QA F1 lands in #3 |
| Summary LLM cost on long docs | Haiku + prompt cache; skip small/table nodes; flag to disable in eval |
| Refactor breaks existing loader tests | move logic verbatim first, refactor behind green tests (TDD) |

---

## 11. Build sequence (slices for the implementation plan)

1. **Refactor to `Extractor` protocol** — extract current logic into `DoclingStandardExtractor`, prove with stub-swap test, keep all tests green. (No behavior change.)
2. **`DoclingVLMExtractor`** — new default, with fallback. Record block-extraction A/B vs standard.
3. **`TreeBuilder` (heading path) + models** — golden-tree fixtures.
4. **RAPTOR fallback** (`tree_raptor.py`) — bottom-up cluster+summarize for headingless docs; shape/coverage tests.
5. **Node summaries** — Haiku + caching, optional flag.
6. **MMLongBench-Doc eval harness** — loader + ingestion gates + page Recall@k baseline; `make eval-mmlb`; numbers into `docs/benchmarks.md`.
7. **200+ page async path** — wire batched VLM into existing job API; OOM/throughput check.

---

## 12. Research grounding & corrections (2026-06-11)

One research iteration (22 sources, 25 claims verified 3-vote, 0 refuted) grounded this spec before implementation. Key primary sources and what they changed:

| Source | Used for | Effect on spec |
|---|---|---|
| [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) + [Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench) | Confirm PageIndex = real vectorless reasoning-tree repo | Cite as the #2 retrieval reference; **98.7% FinanceBench is vendor self-reported (medium confidence), not a baseline to claim we beat** |
| [Docling VLM docs](https://docling-project.github.io/docling/usage/vision_models/) | Pin VLM API | `VlmPipeline`/`VlmPipelineOptions`/`vlm_model_specs`; granite-docling-258M default; **MLX 6.15s/page vs 102.2s Transformers** |
| [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), [IBM announce](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion) | Default VLM model | DocTags output; replaces SmolDocling preview |
| [MMLongBench-Doc GitHub](https://github.com/mayubo2333/MMLongBench-Doc) + [HF dataset](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) + [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ae0e43289bffea0c1fa34633fc608e92-Paper-Datasets_and_Benchmarks_Track.pdf) | Eval anchor | GPT-4o **44.9%** F1; **`evidence_pages` per question confirmed** → page Recall@k viable; license to-verify |
| [RAPTOR, arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | Headingless tree build | Added §5a bottom-up cluster+summarize fallback |
| Nemotron-Parse 1.1, Mistral OCR 3, dots.ocr, olmOCR ([landscape](https://arxiv.org/html/2511.20478v1)) | Pluggable engines | Named Nemotron-Parse as concrete second `Extractor`; Mistral OCR as API tier |

---

*Next: implementation plan via writing-plans skill.*
