# ContextIQ Blueprint: 100–300 Page Documents

> **Goal:** Ingest, index, and query 100–300 page enterprise PDFs (SEC 10-Ks, technical reports, compliance manuals) without manual tuning, OOM crashes, or multi-minute query latency.
>
> **Status:** Planning doc — May 2026  
> **Scope:** ContextIQ applied vertical (`projects/contextiq/`)

---

## 1. Executive Summary

ContextIQ already handles **multi-document corpora** with real SEC filings and NASA workbooks. The gap is not "does RAG work?" — it is **scale ergonomics** for a single 100–300 page document:

| Concern | Today | Target |
|---|---|---|
| Ingest 200-page PDF | 15–40 min, OOM risk | **< 5 min** on Mac Mini, no crash |
| Chunks per 200-page doc | ~2,000–6,000 (estimated) | Same, but **hierarchical** |
| Query latency | Loads **entire corpus** into RAM per search | **< 2 s** p95, scoped to doc |
| Retrieval quality on long docs | Good on seed set; context fragmentation risk | **Recall@20 ≥ 0.85** on 300-page eval |
| Operator experience | Works if you wait and pray | **Effortless** — upload, ask, cite |

**Core strategy:** Don't embed whole documents. Don't scan whole corpora per query. Use **hierarchical parent-child chunks**, **contextual embeddings**, **Qdrant hybrid search with payload filters**, and **batched/streaming Docling ingestion**.

---

## 2. Where We Are Today

### What works

- Docling parsing → structural blocks (headings, tables, figures)
- Structure-aware chunking (text windows, table row windows, figure atomicity)
- Hybrid retrieval pipeline: vector + lexical + section/financial anchors + expansion + rerank
- Token-budgeted context packets with citation metadata
- Grounded answer synthesis (Anthropic + extractive fallback)
- FastAPI dashboard + CLI
- 87 tests passing; 12-query retrieval eval with content anchors

### What breaks at 100–300 pages

#### A. Ingestion (Docling)

Current loader (`ingestion/loader.py`):

- `generate_page_images=True`, `generate_picture_images=True`
- Picture classification + description enrichment **on by default**
- Single-pass conversion of entire PDF
- No page-range batching, no progress streaming

**Observed industry pain** ([Docling #2892](https://github.com/docling-project/docling/issues/2892), [#3345](https://github.com/docling-project/docling/issues/3345)):

- 180-page image-rich PDF → **20+ minutes**
- 700-page PDF → **OOM ~page 300–345** with default C++ backend
- Picture enrichment and sequential VLM inference dominate cost

#### B. Storage & memory

`LocalDocumentStore.load_blocks()` loads **every document JSON** on:

- Every vector search (`_search_vector` builds full `by_id` map)
- Every lexical candidate scan (`CandidateGenerator.lexical_candidates` iterates all blocks)
- Every section expansion
- Every `/stats` call

For one 300-page doc (~3–8 MB JSON, thousands of blocks) plus existing corpus, **each query allocates the full corpus in memory**.

#### C. Chunk model

Current chunking is **single-level**:

- 350-word text windows / 40-row table windows
- `parent_block_id` metadata exists but **no parent retrieval**
- No section-level or document-level summary vectors
- No contextual prefix before embedding (Anthropic Contextual Retrieval gap)

#### D. Qdrant usage

Current `VectorIndex` (`retrieval/vector_index.py`):

- Local file-backed Qdrant (`QdrantClient(path=...)`)
- Single dense vector via FastEmbed `BAAI/bge-small-en`
- **No sparse/BM25 index** in Qdrant (lexical scan is in Python over all blocks)
- **No payload indexes** on `document_id`, `page`, `section_path`, `block_type`
- **No grouping** by document/section at query time
- Batch upsert only via `client.add()` — no tuned batch sizes
- Not safe for concurrent API + eval (documented in `code-flow.md`)

#### E. UX

- `/ingest` is synchronous — HTTP request blocks until full parse + index completes
- No job status, no page progress, no resumable ingest

---

## 3. Scale Math (200-Page SEC 10-K)

Rough planning numbers:

| Stage | Estimate |
|---|---|
| Docling blocks (pre-chunk) | 800–2,000 |
| Chunks after `DocumentChunker` | 2,000–6,000 |
| Embedding points in Qdrant | 2,000–6,000 per doc |
| JSON storage | 2–8 MB per doc |
| Vector index size (bge-small, 384d) | ~3–10 MB per doc |
| Contextual prefix tokens (if enabled) | +50–100 tokens × chunk count → one-time LLM cost |

**Query path today:** load 6,000+ blocks × N documents → score in Python → embed query → Qdrant top-k → rerank.

**Query path target:** Qdrant hybrid prefetch (dense + sparse) with `document_id` filter → group by section → expand to parent → pack ≤6k tokens.

---

## 4. Research Synthesis

### Must-steal ideas (ranked by ROI)

| Source | Key idea | ContextIQ application |
|---|---|---|
| [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | Prepend 50–100 token situating context to each chunk before embed + BM25 | **Phase 2** — use Haiku + prompt caching per document |
| [Qdrant Chunking Strategies](https://qdrant.tech/course/essentials/day-1/chunking-strategies/) | Focused chunks + rich payload metadata + filtered retrieval | Already partial — need payload indexes + filters |
| [Qdrant Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/) | Prefetch dense + sparse, fuse with RRF, optional rerank | **Phase 2** — replace Python lexical scan |
| [Qdrant Grouping API](https://qdrant.tech/documentation/search/search/) | Group by `document_id` or `section_id`, best chunk per group | **Phase 2** — avoid 8 chunks from same table |
| [ParentDocumentRetriever pattern](https://python.langchain.com/docs/how_to/parent_document_retriever/) | Small child chunks for search, large parent for context | **Phase 1** — core architecture change |
| [SAKI-RAG (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.63/) | Sentence-level linking + dual-axis retrieval (semantic + contextual relevance) | Inform expansion/rerank — keep sentence adjacency |
| [LongRefiner (ACL 2025)](https://aclanthology.org/2025.acl-long.176/) | Hierarchical document refinement before LLM | Optional Phase 3 — section summarization |
| [MultiDocFusion (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1062.pdf) | DFS over section tree → hierarchical chunks | Align with Docling heading hierarchy |
| [M3DocDep (2026)](https://arxiv.org/html/2605.18774v1) | Recover block dependencies before chunking | Future — cross-page table healing |
| [Docling GPU/perf docs](https://github.com/docling-project/docling/blob/main/docs/usage/gpu.md) | `page_batch_size`, threaded pipeline, disable enrichment | **Phase 1** — immediate ingest wins |

### What NOT to do yet

- Full GraphRAG over 300 pages — overkill for portfolio MVP
- Fine-tune embedding models (SitEmb) — use contextual prefixes first
- Replace Docling with custom LVLM pipeline — too much scope
- Multi-document cross-reference graph — Phase 4 at earliest

---

## 5. Target Architecture

```text
                    ┌─────────────────────────────────────┐
                    │         INGEST (async job)          │
                    └─────────────────────────────────────┘
  PDF 100-300pp ──► Split page batches (50pp)
                         │
                         ▼
                   Docling (fast profile)
                   - pypdfium2 backend option
                   - no picture enrichment by default
                   - table mode FAST for first pass
                         │
                         ▼
              Section tree from headings
              (document → part → section → block)
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     PARENT chunks              CHILD chunks
     (section-level,            (250-400 tokens,
      1.5-3k tokens)            table row windows)
            │                         │
            │                   Contextual prefix
            │                   (Haiku + cache)
            │                         │
            ▼                         ▼
     JSON doc store              Qdrant collection
     (parents)                   (children + vectors)
                                 - dense: bge-small or embed-v4
                                 - sparse: BM25 (Qdrant sparse)
                                 - payload: document_id, page,
                                   section_path, parent_id, type

                    ┌─────────────────────────────────────┐
                    │              QUERY                    │
                    └─────────────────────────────────────┘
  Question ──► QueryAnalyzer (intent, entities)
                    │
                    ▼
         Qdrant Query API (hybrid prefetch + RRF)
         filter: document_id = X (optional)
         group_by: section_id (top 1-2 per section)
                    │
                    ▼
         Expand child → parent section chunk
         SectionExpander (neighbor blocks, table rows)
                    │
                    ▼
         ContextEngine (token budget, trim tables)
                    │
                    ▼
         GroundedAnswerer (Claude + citations)
```

---

## 6. Qdrant Design (Concrete)

### Collection schema

```python
# One collection: contextiq_chunks_v2
vectors_config = {
    "dense": VectorParams(size=384, distance=Distance.COSINE),
    "sparse": SparseVectorParams(modifier=Modifier.IDF),  # BM25
}
```

### Payload fields (all indexed)

| Field | Type | Index | Purpose |
|---|---|---|---|
| `document_id` | keyword | ✅ | Scope search to one 300pp doc |
| `parent_id` | keyword | ✅ | ParentDocument retrieval |
| `section_id` | keyword | ✅ | Grouping / dedup |
| `section_path` | text | ✅ | Full-text filter ("Risk Factors") |
| `page` | integer | ✅ | Page-scoped questions |
| `block_type` | keyword | ✅ | table / figure / text / heading |
| `chunk_level` | keyword | ✅ | `child` \| `parent` \| `summary` |
| `block_id` | keyword | ✅ | Citation anchor |
| `source_path` | keyword | — | Audit trail |

Create indexes via `create_payload_index()` for every filtered field ([Qdrant payload indexing](https://qdrant.tech/documentation/concepts/payload/#payload-indexing)).

### Query pattern

```python
client.query_points(
    collection_name="contextiq_chunks_v2",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=80,
                 filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])),
        Prefetch(query=sparse_vec, using="sparse", limit=80,
                 filter=same_filter),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    group_by="section_id",
    group_size=2,
    limit=20,
)
```

Then map child hits → parent payloads from JSON store (or parent points in Qdrant).

### Ingest performance

- Upsert in batches of **256–512** points
- Use `wait=True` only on final batch per document
- Move from local path mode to **Qdrant server** (Docker) when single-doc points > 5k or concurrent users
- Keep local path for dev/demo; document the concurrency constraint

### Anti-patterns to avoid

- ❌ Embedding parent and child at same granularity
- ❌ Storing full parent text in every child payload (duplication explosion)
- ❌ Python full-corpus lexical scan at query time
- ❌ Single vector per multi-topic section
- ❌ Running ingest + eval + API against same local Qdrant path concurrently

---

## 7. Docling Ingest Profile (Concrete)

### Fast profile (default for >50 pages)

```python
PdfPipelineOptions(
    generate_page_images=False,      # enable only when visual intent detected
    generate_picture_images=False,   # same
    do_picture_classification=False,
    do_picture_description=False,
    images_scale=1.0,
)
# backend: pypdfium2 for speed OR docling-parse for structure quality
# table_structure_options.mode = "FAST"
```

### Quality profile (≤50 pages or user opt-in)

Current behavior — full enrichment, page images, picture descriptions.

### Batched ingestion algorithm

```text
1. Split PDF into page ranges [1-50], [51-100], ...
2. New DocumentConverter per batch (avoid C++ backend memory leak)
3. Parse batch → blocks
4. Merge blocks; detect broken multi-page tables (heuristic: trailing incomplete row)
5. Chunk → contextualize → upsert batch to Qdrant
6. Emit progress: pages_done / total_pages
7. Concatenate DoclingDocument if using docling merge API
```

Reference: [Docling #2892](https://github.com/docling-project/docling/issues/2892), [#3345](https://github.com/docling-project/docling/issues/3345), [GPU perf guide](https://github.com/docling-project/docling/blob/main/docs/usage/gpu.md).

### Async job API

```text
POST /ingest/async → { job_id }
GET  /ingest/{job_id} → { status, pages_done, pages_total, blocks_indexed }
```

Store job state in SQLite (`data/jobs.db`). Background worker via FastAPI `BackgroundTasks` or separate `contextiq-worker` CLI.

---

## 8. Parent-Child Chunk Model

### Levels

| Level | Size | Stored in | Indexed in Qdrant |
|---|---|---|---|
| **Child** | 200–400 tokens (~150–300 words) | Qdrant + JSON | ✅ dense + sparse |
| **Parent** | 1,500–3,000 tokens (section) | JSON (`parents/{doc_id}.json`) | Optional summary vector |
| **Document summary** | 300–500 tokens | JSON metadata | Optional (for routing) |

### Child chunk rules (keep existing + adjust)

- Text: **250 words**, overlap **40** (slightly smaller than today for precision)
- Tables: **25 rows**, overlap **2** (SEC tables are wide — smaller rows per chunk)
- Figures: atomic (unchanged)
- Headings: never split; attach to following content in parent

### Parent definition

- Parent = contiguous blocks under same **H1/H2 section path** from Docling
- If section > 3k tokens, split at H3 boundaries
- Parent ID = `{document_id}:{section_slug}`

### Retrieval flow

1. Search **child** vectors (hybrid)
2. Resolve `parent_id` from payload
3. Fetch parent text from doc store (not re-embedded whole)
4. If parent too large for budget, use child hits + `SectionExpander` neighbors

This implements the [ParentDocumentRetriever](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/parent_document_retriever.py) pattern without LangChain dependency.

---

## 9. Contextual Retrieval (Phase 2)

Per [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval):

> Prepend chunk-specific explanatory context (~50–100 tokens) before embedding and BM25 indexing.

### Prompt template (per chunk)

```text
Document: {document_title} ({page_range})
Section: {section_path}
<chunk>{chunk_text}</chunk>
Situate this chunk for retrieval. Output ONLY a short context prefix (2-3 sentences).
```

### Cost control

- Use **Claude Haiku** with **prompt caching** — full document text cached once per doc
- Batch 50 chunks per API call where possible
- Store `context_prefix` in payload; embed `prefix + chunk_text`
- Skip for tables (embed row headers + section path only)

Reported improvement: **35–49% fewer retrieval failures**; **67% with reranking**.

---

## 10. Implementation Phases

### Phase 1 — Effortless ingest (2 weeks)

**Outcome:** 200-page PDF ingests reliably in <5 min on Mac Mini.

| Task | Files |
|---|---|
| Docling fast/quality profiles | `ingestion/loader.py`, `core/config.py` |
| Page-batch ingest with progress | `ingestion/batch.py` (new), `api/main.py` |
| Async ingest job + SQLite status | `jobs/` (new) |
| Disable picture enrichment >50 pages | `loader.py` |
| Ingest CLI: `contextiq ingest --async large.pdf` | `cli/app.py` |

**Exit criteria:**

- [ ] 200-page SEC PDF completes without OOM
- [ ] Progress visible in CLI and API
- [ ] Existing 87 tests still pass

### Phase 2 — Scalable retrieval (2–3 weeks)

**Outcome:** Query never loads full corpus; hybrid search in Qdrant.

| Task | Files |
|---|---|
| Parent-child chunk model | `ingestion/chunking.py`, `ingestion/hierarchy.py` (new) |
| Qdrant v2 collection (dense + sparse) | `retrieval/vector_index.py` |
| Payload indexes + document filter | `retrieval/vector_index.py` |
| Replace Python lexical with Qdrant sparse | `retrieval/candidates.py` |
| Group-by section in query | `retrieval/pipeline.py` |
| Per-document block cache / lazy load | `retrieval/store.py` |
| Parent resolution after child hit | `retrieval/parent_resolver.py` (new) |

**Exit criteria:**

- [ ] Query with `document_id` filter: p95 < 2s on 6k-chunk doc
- [ ] Memory flat regardless of corpus size (scoped queries)
- [ ] Recall@20 ≥ 0.80 on expanded qrels (include 300pp doc)

### Phase 3 — Retrieval quality (2 weeks)

**Outcome:** Long-doc questions don't lose cross-chunk context.

| Task | Files |
|---|---|
| Contextual prefix generation | `ingestion/contextualizer.py` (new) |
| Cross-encoder reranker (mini cross-encoder) | `retrieval/reranker.py` (new) |
| Multi-page table heal heuristic | `ingestion/merge.py` (new) |
| Expand qrels to 30+ queries on 300pp doc | `tests/evals/qrels/` |
| Wire ContextForge for budget pack (optional) | integration with `projects/contextforge/` |

**Exit criteria:**

- [ ] Recall@20 ≥ 0.85 on 300-page eval set
- [ ] Manual demo Q1–Q12 from `demo-questions.md` pass on 300pp doc
- [ ] Contextual ablation shows ≥15% recall lift

### Phase 4 — Production polish (1 week)

| Task | Notes |
|---|---|
| Qdrant Docker Compose service | Replace local path for demo |
| CI benchmark gate | `make eval-long-doc` in GitHub Actions |
| README benchmark table for 300pp | Portfolio signal |
| Demo video | 3 questions on 200+ page doc |

---

## 11. Evaluation Plan

### New eval: `long_doc_300pp.json`

Add qrels for one **full 300-page** document (recommend: a single SEC 10-K or NASA report):

| Category | # queries | Example |
|---|---:|---|
| Exact table lookup | 5 | "What was total revenue in FY2025?" |
| Section synthesis | 5 | "Summarize risk factors with citations" |
| Cross-section | 5 | "Compare MD&A outlook vs risk disclosures" |
| Figure/chart | 3 | "What does the revenue chart show?" |
| Negative / missing | 3 | "What penalty for late payment?" (should refuse) |
| Page-specific | 4 | "What appears on page 142?" |

### Metrics

| Metric | Baseline (today) | Phase 2 target | Phase 3 target |
|---|---:|---:|---:|
| Ingest time (200pp) | ~20+ min | < 5 min | < 4 min |
| Query p95 latency | unbounded | < 2 s | < 1.5 s |
| Recall@20 (300pp qrels) | TBD | ≥ 0.80 | ≥ 0.85 |
| Peak RAM at query | O(corpus) | O(doc) | O(doc) |
| Answer faithfulness (LLM judge) | TBD | ≥ 0.80 | ≥ 0.88 |

### Benchmark commands (target)

```bash
make ingest-bench FILE=data/raw/large-300pp.pdf   # time + memory report
make eval-long-doc                                 # 300pp qrels
make eval-retrieval                                # existing seed (regression)
```

---

## 12. Config Additions

```yaml
# config/ingest_profile.yaml (new)
profiles:
  fast:
    max_pages_before_batch: 50
    page_batch_size: 50
    picture_enrichment: false
    generate_page_images: false
    pdf_backend: pypdfium2
    table_mode: FAST
  quality:
    picture_enrichment: true
    generate_page_images: true
    pdf_backend: docling_parse
    table_mode: ACCURATE

# config/chunk_profile.yaml (new)
chunking:
  child:
    max_text_words: 250
    text_overlap_words: 40
    max_table_rows: 25
    table_overlap_rows: 2
  parent:
    max_tokens: 3000
    split_at_heading_level: 3
```

---

## 13. Relationship to ContextForge

| Layer | Owner | Notes |
|---|---|---|
| Retrieval (child search) | ContextIQ | Qdrant hybrid + hierarchy |
| Context packing / compression | ContextForge | Integrate in Phase 3 — `ContextEngine` delegates budget trim |
| Eval harness | Shared | ContextIQ qrels + ContextForge evidence metrics |

Keep projects separate; integrate via optional `contextforge` dependency in Phase 3.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Docling batch splits break multi-page tables | Table boundary heuristic + manual heal list |
| Contextual prefix LLM cost on 6k chunks | Prompt caching; skip tables; batch API calls |
| Local Qdrant OOM at 50k+ points | Docker Qdrant server; HNSW `m` tuning |
| pypdfium2 reduces structure quality | Auto-select backend by page count + doc type |
| Scope creep into agents/MCP | This blueprint is retrieval/ingest only |

---

## 15. Recommended Starting Point

**Start Phase 1 this week.** The single highest-leverage change is **batched Docling ingest with fast profile** — everything else is moot if a 200-page upload crashes or takes 30 minutes.

Immediate quick wins (1–2 days):

1. Add `enable_picture_enrichment=False` default for PDFs > 50 pages
2. Add page-batch ingest loop with progress logging
3. Add `document_id` filter to vector search (even before full Qdrant hybrid)
4. Lazy-load blocks: `load_blocks(document_id=...)` instead of full corpus

---

## References

1. Anthropic — [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Sep 2024)
2. Anthropic — [Prompt Caching](https://www.anthropic.com/news/prompt-caching) (Dec 2024)
3. Qdrant — [Text Chunking Strategies](https://qdrant.tech/course/essentials/day-1/chunking-strategies/)
4. Qdrant — [Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/)
5. Qdrant — [Search & Grouping](https://qdrant.tech/documentation/search/search/)
6. Qdrant — [Payload Indexing](https://qdrant.tech/documentation/concepts/payload/#payload-indexing)
7. SAKI-RAG — [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.63.pdf)
8. LongRefiner — [ACL 2025](https://aclanthology.org/2025.acl-long.176.pdf)
9. MultiDocFusion — [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1062.pdf)
10. M3DocDep — [arXiv 2026](https://arxiv.org/html/2605.18774v1)
11. SitEmb-v1.5 — [arXiv 2025](https://arxiv.org/html/2508.01959v2)
12. Docling — [GPU/perf guide](https://github.com/docling-project/docling/blob/main/docs/usage/gpu.md)
13. Docling — [Large PDF issues #2892, #3345](https://github.com/docling-project/docling/issues/2892)

---

*Next action: pick Phase 1 task list and implement batched ingest.*
