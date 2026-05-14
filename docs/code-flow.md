# ContextIQ Code Flow

This document maps the current code path by function. It is the working reference for the next refactor.

## 1. CLI Entry Points

`src/contextiq/cli/app.py`

- `ingest(path, index=True)`
  - Calls `DocumentLoader.load(path)`.
  - Saves parsed blocks with `LocalDocumentStore.save_blocks(blocks)`.
  - Optionally indexes vectors with `LocalDocumentStore.index_blocks(blocks)`.

- `ask(question)`
  - Creates `LocalDocumentStore()`.
  - Creates `ContextEngine(store)`.
  - Calls `ContextEngine.build_context(question)`.
  - Prints `ContextPacket.to_markdown()`.

- `inspect_context()`
  - Calls `LocalDocumentStore.stats()`.

- `eval_retrieval(qrels, limit, k)`
  - Loads qrels with `load_qrels(qrels)`.
  - Runs `run_retrieval_eval(...)`.
  - Uses `LocalDocumentStore.search(question, limit)` as the retriever.
  - Scores exact block-ID qrels and stable content anchors.

## 2. API Entry Points

`src/contextiq/api/main.py`

- `create_app(store_path=None)`
  - Builds the FastAPI app and closes over `store()`.

- `POST /ingest -> ingest_document(file, index=True)`
  - Writes uploaded file to `data/raw`.
  - Calls `DocumentLoader().load(destination)`.
  - Calls `LocalDocumentStore.save_blocks(blocks)`.
  - Optionally calls `LocalDocumentStore.index_blocks(blocks)`.

- `POST /context -> build_context(request)`
  - Calls `ContextEngine(store, token_budget).build_context(question, limit)`.
  - Returns markdown plus source metadata.

## 3. Gradio UI Flow

`src/contextiq/ui/gradio_app.py`

- `ingest_file(file_path, backend_url, build_index)`
  - Sends file to FastAPI `POST /ingest`.

- `ask_question(question, backend_url, token_budget, limit)`
  - Sends question to FastAPI `POST /context`.
  - Displays the context packet markdown.

- `load_stats(backend_url)`
  - Calls FastAPI `GET /stats`.

## 4. Ingestion Flow

`src/contextiq/ingestion/loader.py`

- `DocumentLoader.load(path)`
  - Verifies the file exists.
  - Routes Excel workbooks to `_load_workbook(path)`.
  - Tries `_load_with_docling(path)`.
  - If Docling fails:
    - Raises when `strict_docling=True`.
    - Otherwise falls back to `_load_plain_text(path, parser_error=...)`.
  - Sends parsed blocks through `DocumentChunker.chunk_blocks(...)`.

`src/contextiq/ingestion/chunking.py`

- `DocumentChunker.chunk_blocks(blocks)`
  - Preserves already-focused blocks.
  - Keeps figures and headings atomic.
  - Splits oversized text blocks with overlapping word windows.
  - Splits oversized markdown tables with overlapping row windows.

- `ChunkingConfig`
  - `max_text_words=350`
  - `text_overlap_words=50`
  - `max_table_rows=40`
  - `table_overlap_rows=3`

Chunk metadata follows Qdrant chunking guidance: each vector carries enough payload to group, filter, cite, and inspect the result. Important fields include `parent_block_id`, `chunk_index`, `chunk_count`, `chunk_strategy`, row/word ranges, section path, page, parser, and sheet/table metadata.

- `_load_with_docling(path)`
  - Builds a Docling `DocumentConverter` with `PdfPipelineOptions`.
  - Enables page and picture image generation for PDFs when Docling supports it.
  - Requests Docling picture classification and picture description enrichment.
  - If enriched conversion fails, retries with basic Docling page/picture image
    generation so ingestion remains reliable.
  - Passes the Docling document to `_load_docling_document(...)`.

- `_load_docling_document(document, path)`
  - Iterates `document.iterate_items()`.
  - Converts Docling labels with `_block_type_from_label(label)`.
  - Extracts text/table/figure content with `_text_from_item(...)`.
  - Preserves figures as `BlockType.FIGURE` with caption text and visual metadata.
  - Preserves tables as `BlockType.TABLE` with markdown text and visual metadata.
  - Saves Docling visual crops under `data/processed/visuals/...` when the
    parser exposes an image for the table or figure item.
  - Stores Docling provenance, bounding boxes, pages, parser labels, visual
    artifact paths, visual class predictions, and visual descriptions in block
    metadata.
  - Extracts page numbers with `_page_from_item(item)`.
  - Emits `DocumentBlock` objects with parser metadata.

- `_load_plain_text(path, parser_error=None)`
  - Reads markdown/text.
  - Calls `_split_markdown(...)`.
  - Emits `DocumentBlock` objects with `parser=plain_text` metadata.

- `_load_workbook(path)`
  - Reads `.xlsx/.xlsm/.xltx/.xltm` files with `openpyxl`.
  - Emits one raw `BlockType.TABLE` block per non-empty sheet.
  - Uses the sheet name as `section_path`.
  - Renders sheet rows as markdown tables for lexical/vector retrieval.
  - Large sheets are then split by `DocumentChunker` into row-window chunks.

## 5. Local Storage Flow

`src/contextiq/retrieval/store.py`

- `LocalDocumentStore.save_blocks(blocks)`
  - Loads existing blocks using `load_blocks()`.
  - Replaces blocks for incoming `document_id`s.
  - Writes one JSON file per document under `data/processed/documents/`.
  - Writes `data/processed/blocks.json` as a manifest of document IDs.
  - Preserves legacy list-of-blocks data when migrating to manifest mode.

- `LocalDocumentStore.load_blocks()`
  - If `blocks.json` is a manifest, loads document files from `documents/`.
  - If `blocks.json` is legacy block JSON, loads it directly.

- `LocalDocumentStore.stats()`
  - Counts loaded documents and blocks.

## 6. Vector Index Flow

`src/contextiq/retrieval/vector_index.py`

- `VectorIndex.index_blocks(blocks)`
  - Deletes existing vectors for incoming document IDs using `_delete_existing_documents(...)`.
  - Adds block texts to Qdrant/FastEmbed with metadata payloads.
  - Uses `_point_id(block_id)` to convert citation IDs into deterministic UUIDs.

- `_delete_existing_documents(document_ids)`
  - Ignores missing collections.
  - Raises on unexpected delete failures to avoid stale vector state.

- `VectorIndex.search(query, limit)`
  - Calls Qdrant local FastEmbed query.
  - Returns original `block_id`s from metadata.

Local Qdrant caveat: `QdrantClient(path=...)` is not safe for concurrent local access. Do not run the API server, vector-index tests, and retrieval eval against the same `data/qdrant` path at the same time. Production mode should move to Qdrant server or a managed Qdrant instance.

## 7. Retrieval Flow

`src/contextiq/retrieval/store.py`

- `LocalDocumentStore.search(query, limit)`
  - Delegates to `RetrievalPipeline.search(query, limit)`.
  - Keeps storage, manifest handling, and vector-index access separate from retrieval orchestration.

- `LocalDocumentStore.search_with_trace(query, limit)`
  - Delegates to `RetrievalPipeline.search_with_trace(query, limit)`.
  - Returns ranked `RetrievalHit` objects with stages, score, and reason.

`src/contextiq/retrieval/query.py`

- `QueryAnalyzer.analyze(query)`
  - Normalizes raw tokens with `normalize_term(...)`.
  - Detects `QueryIntent.GENERAL` or `QueryIntent.FINANCIAL_PERFORMANCE`.
  - Produces `QueryAnalysis(raw_terms, terms, intent)`.
  - Adds explicit expansions for legal, regulatory, risk, contract, and financial terms.

`src/contextiq/retrieval/candidates.py`

- `CandidateGenerator.generate(query, limit)`
  - Gets vector candidates through the injected `vector_search(query, limit)` callable.
  - Gets lexical candidates with `lexical_candidates(...)`.
  - Gets heading anchors with `section_anchor_candidates(...)`.
  - Adds financial section anchors with `financial_anchor_candidates(...)` for `QueryIntent.FINANCIAL_PERFORMANCE`.
  - Deduplicates first-pass candidates by `block_id`.

- `CandidateGenerator.generate_with_trace(query, limit)`
  - Same retrieval behavior as `generate(...)`.
  - Tags each candidate with source stages such as `vector`, `lexical`, `section_anchor`, and `financial_anchor`.

- `CandidateGenerator.lexical_candidates(query, limit)`
  - Scores normalized query-term overlap across block text and section path.
  - Deprioritizes headings when evidence text has equal lexical overlap.

`src/contextiq/retrieval/expansion.py`

- `SectionExpander.expand(candidates, limit, query=None)`
  - Adds same-section heading prefixes for evidence blocks.
  - Adds same-section following neighbors for evidence continuity.
  - Expands heading candidates into nearby section content.
  - Uses `CandidateRanker.heading_expansion_window(...)` for query-aware section windows.

- `SectionExpander.expand_with_trace(candidates, limit, query=None)`
  - Preserves first-pass retrieval stages.
  - Tags neighbor evidence with `expansion`.

`src/contextiq/retrieval/ranker.py`

- `CandidateRanker.rerank(query, candidates)`
  - Deduplicates candidates by `block_id`.
  - Scores each candidate with `score(...)`.

- `CandidateRanker.score(analysis, block)`
  - Scores text/section term overlap.
  - Applies block-type bonuses/penalties.
  - Applies financial-performance bonuses and risk/noise penalties.

- `CandidateRanker.heading_expansion_window(query, heading)`
  - Controls how much evidence to pull after a matched heading.
  - Gives larger windows to dense financial sections.

- `CandidateRanker.apply_intent_precision(query, candidates)`
  - Keeps financial-performance answers focused on section evidence rather than generic risk prose.

`src/contextiq/retrieval/pipeline.py`

- `RetrievalPipeline.search(query, limit)`
  - Returns final ranked `DocumentBlock` objects for legacy callers.

## 8. Retrieval Eval Flow

`src/contextiq/evals/retrieval.py`

- `load_qrels(path)`
  - Reads `tests/evals/qrels/retrieval_seed.json`.
  - Supports `relevant_blocks` for exact block IDs.
  - Supports `relevant_anchors` for stable content targets.

- `ContentAnchor`
  - Defines evidence requirements by `text_contains`, `source_contains`, `document_id`, and `block_type`.
  - Lets evals survive Docling parse shifts, row-window chunking, and deterministic ID changes.

- `run_retrieval_eval(cases, retrieve, limit, k)`
  - Calls the retriever once per question.
  - Scores ranked `DocumentBlock` objects with `evaluate_ranked_blocks(...)`.

- `evaluate_ranked_blocks(cases, ranked_by_query_id, k)`
  - Counts a hit when a block ID matches or when a block satisfies a content anchor.
  - Reports Recall@k, Precision@k, MRR, and NDCG@k.

- `RetrievalPipeline.search_with_trace(query, limit)`
  - Computes `candidate_limit`.
  - Calls `CandidateGenerator.generate_with_trace(...)`.
  - Calls `SectionExpander.expand_with_trace(...)`.
  - Calls `CandidateRanker.rerank(...)`.
  - Calls `CandidateRanker.apply_intent_precision(...)`.
  - Returns final ranked `RetrievalHit` objects with `rank`, `score`, `stages`, and `reason`.

Multimodal retrieval note: image/chart retrieval does not require blind LLM
search. Figures are represented as retrievable text blocks using captions,
Docling visual descriptions, visual class predictions, and visual metadata. The
LLM answer layer explains retrieved visual evidence with citations and must
state when visual evidence is only partially described.

## 8. Context Assembly Flow

`src/contextiq/context/engine.py`

- `ContextEngine.build_context(question, limit)`
  - Calls `store.search(question, limit)`.
  - Counts tokens with `tiktoken`.
  - Adds blocks until `token_budget` is reached.
  - Emits `ContextPacket`.

`src/contextiq/context/models.py`

- `ContextPacket.to_markdown()`
  - Renders question, budget, token usage, dropped candidates, and sources.

## 9. Evaluation Flow

`src/contextiq/evals/retrieval.py`

- `load_qrels(path)`
  - Loads graded query relevance judgments from JSON.

- `run_retrieval_eval(cases, retrieve, limit, k)`
  - Calls the provided retriever for each query.
  - Converts retrieved blocks into ranked block IDs.
  - Calls `evaluate_ranked_ids(...)`.

- `evaluate_ranked_ids(cases, ranked_by_query_id, k)`
  - Calculates Recall@k, Precision@k, MRR, and NDCG@k.
  - Returns `RetrievalEvalReport`.

- `RetrievalEvalReport.to_markdown()`
  - Renders a portfolio-friendly report.

## 10. Current Reliability Baseline

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run contextiq eval-retrieval --limit 20 --k 10
```

Current retrieval baseline:

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.753 |
| Precision@10 | 0.283 |
| MRR | 0.822 |
| NDCG@10 | 0.700 |
