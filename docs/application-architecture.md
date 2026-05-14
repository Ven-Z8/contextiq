# ContextIQ Application Architecture

This document maps the current ContextIQ codebase into a visual flow. It is meant to be reviewed and corrected as the product evolves.

## High-Level System

```mermaid
flowchart TD
    User["User / Portfolio Demo"] --> UI["Browser Dashboard<br/>src/contextiq/api/main.py<br/>GET /"]
    User --> CLI["CLI<br/>src/contextiq/cli/app.py"]
    User --> Gradio["Optional Gradio UI<br/>src/contextiq/ui/gradio_app.py"]

    UI --> API["FastAPI App<br/>create_app()"]
    Gradio --> API
    CLI --> Loader["DocumentLoader.load()"]
    CLI --> Engine["ContextEngine.build_context()"]
    CLI --> Eval["run_retrieval_eval()"]

    API --> IngestEndpoint["POST /ingest<br/>ingest_document()"]
    API --> ContextEndpoint["POST /context<br/>build_context()"]
    API --> AnswerEndpoint["POST /answer<br/>answer_question()"]
    API --> VisualEndpoint["GET /visuals<br/>visual_artifact()"]
    API --> StatsEndpoint["GET /stats<br/>stats()"]

    IngestEndpoint --> Loader
    Loader --> Chunker["DocumentChunker.chunk_blocks()"]
    Chunker --> Store["LocalDocumentStore<br/>save_blocks(), load_blocks()"]
    Store --> JsonStore["JSON Corpus<br/>data/processed/blocks.json<br/>data/processed/documents/*.json"]
    Store --> VectorIndex["VectorIndex.index_blocks()<br/>Qdrant + FastEmbed"]
    VectorIndex --> Qdrant["Local Qdrant Store<br/>data/qdrant"]
    Loader --> Visuals["Visual Artifacts<br/>data/processed/visuals/*.png"]
    VisualEndpoint --> Visuals

    ContextEndpoint --> Engine
    AnswerEndpoint --> Engine
    Engine --> Store
    Store --> Pipeline["RetrievalPipeline.search_with_trace()"]
    Pipeline --> Generator["CandidateGenerator.generate_with_trace()"]
    Pipeline --> Expander["SectionExpander.expand_with_trace()"]
    Pipeline --> Ranker["CandidateRanker.rerank()<br/>apply_intent_precision()"]
    Generator --> QueryAnalyzer["QueryAnalyzer.analyze()"]
    Ranker --> QueryAnalyzer
    QueryAnalyzer --> Profile["RetrievalProfile.default()<br/>config/retrieval_profile.json"]
    Generator --> VectorIndex
    Generator --> JsonStore

    Engine --> Packet["ContextPacket.to_markdown()"]
    AnswerEndpoint --> Answerer["GroundedAnswerer.answer()"]
    Answerer --> Prompt["load_answer_prompt()<br/>prompts/answer.yaml"]
    Answerer --> LLMClient["AnthropicLLMClient.generate()<br/>or ExtractiveFallbackClient.generate()"]
    LLMClient --> Claude["Anthropic Claude API<br/>when ANTHROPIC_API_KEY exists"]
    Answerer --> AnswerResponse["Grounded Answer + Sources + Warnings"]
```

## Ingestion Flow

```mermaid
sequenceDiagram
    participant U as User/UI/CLI
    participant API as FastAPI ingest_document()
    participant Loader as DocumentLoader.load()
    participant Docling as Docling Converter
    participant Chunker as DocumentChunker.chunk_blocks()
    participant Store as LocalDocumentStore.save_blocks()
    participant Vec as VectorIndex.index_blocks()
    participant Disk as data/processed + data/qdrant

    U->>API: Upload document or run contextiq ingest
    API->>Loader: load(path)
    alt Excel workbook
        Loader->>Loader: _load_workbook()
        Loader->>Loader: _rows_to_markdown_table()
    else Markdown/Text
        Loader->>Loader: _load_plain_text()
        Loader->>Loader: _split_markdown()
    else PDF/DOCX/complex document
        Loader->>Docling: _load_with_docling()
        Docling-->>Loader: Docling document items
        Loader->>Loader: _load_docling_document()
        Loader->>Loader: _visual_metadata()
        Loader->>Loader: _save_visual_image()
        Loader->>Loader: _docling_picture_metadata()
    end
    Loader-->>Chunker: list[DocumentBlock]
    Chunker->>Chunker: _chunk_text() / _chunk_table()
    Chunker-->>Store: citation-preserving chunks
    Store->>Disk: write per-document JSON + manifest
    opt Build vector index
        Store->>Vec: index_blocks(blocks)
        Vec->>Vec: _point_id(block_id) as UUID
        Vec->>Disk: Qdrant collection contextiq_blocks
    end
```

## Retrieval And Answer Flow

```mermaid
sequenceDiagram
    participant U as User/UI/CLI
    participant API as /context or /answer
    participant Engine as ContextEngine.build_context()
    participant Store as LocalDocumentStore.search_with_trace()
    participant Pipeline as RetrievalPipeline.search_with_trace()
    participant Analyzer as QueryAnalyzer.analyze()
    participant Gen as CandidateGenerator
    participant Vec as VectorIndex.search()
    participant Exp as SectionExpander
    participant Rank as CandidateRanker
    participant Packet as ContextPacket
    participant Answerer as GroundedAnswerer
    participant LLM as AnthropicLLMClient or Fallback

    U->>API: Ask question + source limit + token budget
    API->>Engine: build_context(question, limit)
    Engine->>Analyzer: analyze(question)
    Engine->>Store: search_with_trace(question, limit)
    Store->>Pipeline: search_with_trace()
    Pipeline->>Gen: generate_with_trace()
    Gen->>Analyzer: detect intent, codes, assets, visual terms
    alt Structured code query
        Gen->>Gen: structured_code_candidates()
    else Asset mapping query
        Gen->>Gen: asset_mapping_candidates()
    else General / financial / visual query
        Gen->>Vec: vector_search()
        Gen->>Gen: lexical_candidates()
        Gen->>Gen: section_anchor_candidates()
        Gen->>Gen: visual_candidates()
        Gen->>Gen: financial_metric_table_candidates()
        Gen->>Gen: financial_anchor_candidates()
    end
    Pipeline->>Exp: expand_with_trace()
    Exp->>Exp: add same-section context or heading windows
    Pipeline->>Rank: rerank()
    Rank->>Rank: score()
    Pipeline->>Rank: apply_intent_precision()
    Pipeline-->>Store: list[RetrievalHit]
    Store-->>Engine: traced hits
    Engine->>Engine: trim exact structured-code or asset rows
    Engine->>Engine: enforce token budget
    Engine-->>Packet: ContextPacket
    alt /context
        API-->>U: ContextResponse with markdown + sources
    else /answer
        API->>Answerer: answer(packet)
        Answerer->>Packet: to_markdown()
        Answerer->>LLM: generate(system_prompt, user_prompt)
        LLM-->>Answerer: grounded answer or extractive fallback
        API-->>U: AnswerResponse with answer + context + warnings
    end
```

## Main Files And Responsibilities

| Area | File | Key classes/functions | Responsibility |
|---|---|---|---|
| API | `src/contextiq/api/main.py` | `create_app()`, `ingest_document()`, `build_context()`, `answer_question()`, `_context_response()`, `_resolve_visual_artifact()` | FastAPI backend, lightweight dashboard, upload, context, answer, stats, visual artifact serving. |
| CLI | `src/contextiq/cli/app.py` | `ingest()`, `ask()`, `inspect_context()`, `eval_retrieval()` | Terminal workflow for ingest, retrieval smoke tests, corpus stats, retrieval evals. |
| Ingestion | `src/contextiq/ingestion/loader.py` | `DocumentLoader.load()`, `_load_with_docling()`, `_load_docling_document()`, `_load_workbook()`, `_visual_metadata()`, `_save_visual_image()` | Converts PDFs, Office-like docs, markdown/text, and Excel into `DocumentBlock`s with sections, pages, tables, figures, visual metadata, and image artifacts. |
| Chunking | `src/contextiq/ingestion/chunking.py` | `DocumentChunker.chunk_blocks()`, `_chunk_text()`, `_chunk_table()` | Splits long text and large tables with overlap while preserving parent metadata. |
| Data model | `src/contextiq/ingestion/models.py` | `BlockType`, `DocumentBlock` | Common block schema for text, headings, tables, and figures. |
| Store | `src/contextiq/retrieval/store.py` | `LocalDocumentStore.save_blocks()`, `load_blocks()`, `search_with_trace()`, `index_blocks()` | JSON-backed corpus store plus retrieval pipeline wiring. |
| Vector index | `src/contextiq/retrieval/vector_index.py` | `VectorIndex.index_blocks()`, `search()`, `_point_id()` | Qdrant/FastEmbed semantic index with deterministic UUID point ids. |
| Query analysis | `src/contextiq/retrieval/query.py` | `QueryAnalyzer.analyze()`, `extract_structured_codes()`, `extract_asset_modes()`, `extract_visual_terms()` | Converts user question into intent, normalized terms, exact codes, asset modes, asset terms, visual intent. |
| Retrieval profile | `src/contextiq/retrieval/profile.py` | `RetrievalProfile.default()`, `from_mapping()` | Loads corpus/domain vocabulary from `config/retrieval_profile.json` instead of hardcoding it in code. |
| Candidate generation | `src/contextiq/retrieval/candidates.py` | `CandidateGenerator.generate_with_trace()`, `structured_code_candidates()`, `asset_mapping_candidates()`, `visual_candidates()`, `lexical_candidates()` | First-pass retrieval from exact structured signals, vector search, lexical search, section anchors, visual evidence, and financial tables. |
| Expansion | `src/contextiq/retrieval/expansion.py` | `SectionExpander.expand_with_trace()` | Adds nearby heading/section context around retrieved blocks. |
| Ranking | `src/contextiq/retrieval/ranker.py` | `CandidateRanker.rerank()`, `score()`, `apply_intent_precision()`, `structured_code_bonus()`, `asset_mapping_bonus()`, `visual_evidence_bonus()` | Scores candidates, filters irrelevant docs, applies intent-specific precision gates. |
| Context engine | `src/contextiq/context/engine.py` | `ContextEngine.build_context()`, `_trim_structured_code_table()`, `_trim_asset_mapping_table()` | Builds token-budgeted context packets and trims large tables down to the exact relevant rows. |
| Context model | `src/contextiq/context/models.py` | `ContextPacket.to_markdown()`, `_format_visual_metadata()` | Serializes evidence, trace, page/section, token counts, visual metadata, and source text. |
| LLM synthesis | `src/contextiq/llm/answerer.py` | `GroundedAnswerer.answer()`, `_default_client()`, `_generate_safely()`, `_grounding_warnings()` | Turns context packet into grounded answer using Anthropic or safe fallback. |
| LLM client | `src/contextiq/llm/client.py` | `AnthropicLLMClient.generate()`, `ExtractiveFallbackClient.generate()` | Provider boundary for answer generation. |
| Prompt | `src/contextiq/llm/prompts.py` and `prompts/answer.yaml` | `load_answer_prompt()`, `AnswerPrompt.render_user()` | Loads the grounded-answer system/user prompt. |
| Eval | `src/contextiq/evals/retrieval.py` | `load_qrels()`, `run_retrieval_eval()`, `evaluate_ranked_blocks()` | Measures recall, precision, MRR, and NDCG against qrels. |

## Data Artifacts

| Artifact | Path | Produced by | Used by |
|---|---|---|---|
| Raw uploads | `data/raw/*` | API `/ingest` or manual copy | `DocumentLoader.load()` |
| Manifest | `data/processed/blocks.json` | `LocalDocumentStore.save_blocks()` | `LocalDocumentStore.load_blocks()` |
| Per-document block JSON | `data/processed/documents/*.json` | `LocalDocumentStore.save_blocks()` | Retrieval pipeline |
| Visual crops | `data/processed/visuals/**/*.png` | `DocumentLoader._save_visual_image()` | API `/visuals`, UI source cards, context markdown |
| Vector DB | `data/qdrant` | `VectorIndex.index_blocks()` | `VectorIndex.search()` |
| Retrieval vocabulary | `config/retrieval_profile.json` | Maintained manually / later auto-profiled | `RetrievalProfile.default()` |
| Answer prompt | `prompts/answer.yaml` | Maintained manually | `GroundedAnswerer.answer()` |
| Eval qrels | `tests/evals/qrels/retrieval_seed.json` | Maintained manually | `contextiq eval-retrieval` |

## Current Retrieval Strategy

```mermaid
flowchart LR
    Q["Question"] --> QA["QueryAnalyzer.analyze()"]
    QA --> Intent{"Intent / Signals"}
    Intent --> Codes["Structured codes<br/>UC/CN/FN/etc from profile"]
    Intent --> Assets["Asset mapping<br/>asset_modes + known_assets"]
    Intent --> Visual["Visual intent<br/>visual_markers"]
    Intent --> Financial["Financial performance"]
    Intent --> General["General"]

    Codes --> ExactCode["structured_code_candidates()"]
    Assets --> AssetRows["asset_mapping_candidates()"]
    Visual --> VisualRows["visual_candidates()"]
    Financial --> FinanceRows["financial_metric_table_candidates()<br/>financial_anchor_candidates()"]
    General --> Hybrid["vector_search()<br/>lexical_candidates()<br/>section_anchor_candidates()"]
    Financial --> Hybrid
    Visual --> Hybrid

    ExactCode --> Expand["SectionExpander<br/>skips expansion for exact structured rows"]
    AssetRows --> Expand
    VisualRows --> Expand
    FinanceRows --> Expand
    Hybrid --> Expand

    Expand --> Rank["CandidateRanker.rerank()<br/>score()"]
    Rank --> Precision["apply_intent_precision()"]
    Precision --> Context["ContextEngine trims + token-packs"]
```

## Portfolio Talking Points

- Docling is used for complex PDFs and visual artifacts: page images, picture images, figure/table metadata, captions, classes, descriptions when Docling provides them.
- Retrieval works without the LLM. The LLM only appears after retrieval, when `/answer` calls `GroundedAnswerer.answer()`.
- Qdrant stores vectors for semantic retrieval, while JSON remains the citation/source-of-truth store.
- `config/retrieval_profile.json` holds demo/domain vocabulary, keeping corpus-specific terms out of core retrieval code.
- The context packet is explicit and inspectable before answer generation, which supports evaluation and groundedness debugging.
