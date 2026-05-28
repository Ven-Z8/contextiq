# ContextIQ: Enterprise RAG Upgrade Plan
> Target: recall@10 ≥ 0.75 on Apple 10-K eval set (current baseline: 0.48)
> Approach: Adaptive chunking + SPLADE neural sparse + ColBERT late-interaction reranking + Qdrant Query API
> Research basis: Qdrant docs (hybrid queries, sparse vectors, multi-vectors, quantization), arxiv:2603.25333 (Adaptive Chunking)

---

## Executive Summary

ContextIQ already has the right skeleton: Qdrant Query API with `prefetch` + RRF fusion, a retrieval pipeline with multiple candidate stages, parent-child hierarchy, and a real eval harness. The current recall@10 = 0.48 ceiling hits three hard walls:

1. **BM25 sparse, not SPLADE** — `Qdrant/bm25` doesn't do term expansion. A query for "Apple operating income" won't match "earnings from operations" even if that's exactly the right block.
2. **No reranking layer** — after RRF fusion, the top-10 go directly to the LLM. There's no second-pass to promote the truly best blocks above near-misses.
3. **One-size chunking** — a 10-K financial table and a paragraph explaining tariff risks get the same chunk strategy. Tables get split mid-row; explanatory prose loses its sentence window.

This plan fixes all three in three sequenced phases, with a fourth phase for production hardening.

---

## Current State Audit

### What's Already Working (Keep It)

```
VectorIndex.search_hybrid()
  prefetch: dense (BAAI/bge-small-en) + BM25 sparse
  fusion:   FusionQuery(RRF)               ← correct approach
  
CandidateGenerator
  stages:   vector | lexical | section_anchor | financial_metric_table
  deduplication, multi-stage trace

HierarchyBuilder + ParentResolver        ← good architecture
ContextEngine with token budget
Eval harness (load_qrels, recall@k, MRR)
```

### What's Broken or Missing

| Gap | Impact | Fix |
|-----|--------|-----|
| BM25 sparse (no term expansion) | Misses synonym/paraphrase queries | → SPLADE neural sparse |
| No reranking after RRF | Wrong blocks promoted by table bias | → ColBERT late interaction |
| Flat chunking strategy | Tables split mid-row; prose loses context window | → Adaptive chunking router |
| CandidateGenerator linear scans in Python | O(N) on all corpus blocks per query | → Push all retrieval into Qdrant |
| Parent text ≠ child text after rebuild | ContentAnchor text_contains breaks | → Anchor matching on original child text |
| "Why/How" queries return tables | Intent blind retrieval | → Block-type-aware query routing |

---

## Phase 1: Adaptive Chunking Engine

### Problem
Enterprise documents like 10-Ks are not uniform. A financial statement table has radically different retrieval needs than a risk factor paragraph. Current chunking treats them identically.

Based on arxiv:2603.25333 (Adaptive Chunking: Optimizing Chunking-Method Selection for RAG), the right approach is to **route each content block through a content-type classifier** and apply the appropriate chunk strategy.

### Content Type → Chunk Strategy Mapping

```
FINANCIAL_TABLE     → keep-whole (max 600 tokens; split by row boundary if over)
NARRATIVE_PARA      → sentence-window (256 tok window, 64 tok stride)
RISK_SECTION        → paragraph-boundary (keep paragraphs intact, max 512 tok)
LIST_ITEMS          → item-level with parent heading (128 tok per item + 64 tok heading prefix)
NUMERICAL_FACT      → anchor-tagged (number + surrounding 100 tok context)
CODE_BLOCK          → keep-whole (never split mid-function)
HEADING             → attach to first N following blocks (section context injection)
```

### Implementation: `contextiq/ingestion/adaptive_chunker.py`

```python
from enum import Enum
from dataclasses import dataclass
from contextiq.ingestion.models import DocumentBlock, BlockType


class ContentProfile(str, Enum):
    FINANCIAL_TABLE = "financial_table"
    NARRATIVE_PARA  = "narrative_para"
    RISK_SECTION    = "risk_section"
    LIST_ITEMS      = "list_items"
    NUMERICAL_FACT  = "numerical_fact"
    HEADING         = "heading"
    GENERIC         = "generic"


class AdaptiveChunker:
    """Route DocumentBlocks to content-appropriate chunking strategies."""

    _FINANCIAL_KEYWORDS = {
        "revenue", "net sales", "operating income", "gross margin",
        "earnings per share", "cash flow", "total assets", "net income",
        "diluted", "fiscal", "quarter", "segment",
    }
    _RISK_KEYWORDS = {"risk", "uncertainty", "may", "could", "adverse", "regulation"}
    _NUMBER_PATTERN = re.compile(r'\$[\d,]+|\d{1,3}(?:,\d{3})+')

    def classify(self, block: DocumentBlock) -> ContentProfile:
        if block.block_type == BlockType.TABLE:
            text_lower = block.text.lower()
            if any(kw in text_lower for kw in self._FINANCIAL_KEYWORDS):
                return ContentProfile.FINANCIAL_TABLE
        if block.block_type == BlockType.HEADING:
            return ContentProfile.HEADING
        text_lower = block.text.lower()
        if any(kw in text_lower for kw in self._RISK_KEYWORDS):
            return ContentProfile.RISK_SECTION
        if any(kw in text_lower for kw in self._FINANCIAL_KEYWORDS):
            return ContentProfile.NUMERICAL_FACT
        if block.text.strip().startswith(("•", "-", "*", "1.", "a.")):
            return ContentProfile.LIST_ITEMS
        if len(block.text.split()) > 80:
            return ContentProfile.NARRATIVE_PARA
        return ContentProfile.GENERIC

    def chunk(self, block: DocumentBlock, profile: ContentProfile) -> list[DocumentBlock]:
        """Return 1+ child blocks appropriate for the content profile."""
        if profile == ContentProfile.FINANCIAL_TABLE:
            return self._chunk_financial_table(block)
        if profile == ContentProfile.NARRATIVE_PARA:
            return self._chunk_sentence_window(block, window=256, stride=64)
        if profile == ContentProfile.RISK_SECTION:
            return self._chunk_by_paragraph(block, max_tokens=512)
        if profile == ContentProfile.LIST_ITEMS:
            return self._chunk_list_items(block)
        if profile == ContentProfile.NUMERICAL_FACT:
            return self._chunk_anchor_window(block, context_tokens=100)
        return [block]  # HEADING, GENERIC: keep as-is
```

### Key Chunking Rules

**Financial Tables** — never split mid-row:
```python
def _chunk_financial_table(self, block: DocumentBlock) -> list[DocumentBlock]:
    rows = block.text.split("\n")
    chunks, current_chunk = [], []
    current_tokens = 0
    for row in rows:
        row_tokens = len(row.split())
        if current_tokens + row_tokens > 600 and current_chunk:
            chunks.append(self._make_child(block, "\n".join(current_chunk)))
            current_chunk, current_tokens = [], 0
        current_chunk.append(row)
        current_tokens += row_tokens
    if current_chunk:
        chunks.append(self._make_child(block, "\n".join(current_chunk)))
    return chunks or [block]
```

**Sentence Windows** (narrative prose):
```python
def _chunk_sentence_window(self, block, window=256, stride=64):
    sentences = self._split_sentences(block.text)
    chunks = []
    start = 0
    while start < len(sentences):
        window_sentences, token_count = [], 0
        for sent in sentences[start:]:
            sent_tokens = len(sent.split())
            if token_count + sent_tokens > window and window_sentences:
                break
            window_sentences.append(sent)
            token_count += sent_tokens
        text = " ".join(window_sentences)
        chunks.append(self._make_child(block, text))
        # Advance by stride (overlap preserves context)
        start += max(1, len(window_sentences) - stride // 15)
    return chunks
```

### Integration Point
`AdaptiveChunker` slots into `BatchIngestor.ingest()` after Docling parsing, before `HierarchyBuilder.build_parents()`. Each block is classified → chunked → tagged with `content_profile` metadata.

---

## Phase 2: SPLADE Neural Sparse Vectors

### Why SPLADE over BM25

| Metric | BM25 | SPLADE | SPLADE++ |
|--------|------|--------|----------|
| MRR@10 (MS MARCO Dev) | 0.184 | 0.322 | 0.368 |
| Term expansion | ✗ | ✓ | ✓ |
| Out-of-vocabulary | fails | expands | expands |
| Memory (1M docs) | ~1 GB | ~1.12 GB | ~1.12 GB |

The key win for enterprise RAG: SPLADE expands "operating income" to match blocks containing "earnings from operations", "profit from operations", etc. This directly addresses the recall gap where financial tables use one vocabulary and the query uses another.

### Collection Schema Upgrade

**Current schema** (implicit via `client.add()`):
```python
vectors_config={"fast-bge-small-en": VectorParams(size=384, distance=COSINE)}
sparse_vectors_config={"fast-sparse-bm25": SparseVectorParams(...)}
```

**Target schema** (explicit, 3 vector types):
```python
from qdrant_client import QdrantClient, models

client.create_collection(
    collection_name="contextiq_v2",
    vectors_config={
        # Stage 1: Dense semantic retrieval
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            on_disk=False,  # Keep in RAM for speed
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        ),
        # Stage 3: ColBERT late-interaction reranking (multi-vector)
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # HNSW disabled — reranking only
        ),
    },
    sparse_vectors_config={
        # Stage 2: Neural sparse (SPLADE) — replaces BM25
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False),
        ),
    },
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,  # Quantized vectors stay in RAM
        )
    ),
)
```

### Embedding Pipeline

**New `EmbeddingPipeline` class:**

```python
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import numpy as np

DENSE_MODEL  = "BAAI/bge-small-en"       # 384 dim, 33M params
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"  # 532 MB, term-expansion
COLBERT_MODEL = "colbert-ir/colbertv2.0"    # 128 dim/token, 0.44 GB


class EmbeddingPipeline:
    """Unified embedding pipeline: dense + SPLADE sparse + ColBERT multivector."""

    def __init__(self) -> None:
        self._dense   = TextEmbedding(DENSE_MODEL)
        self._splade  = SparseTextEmbedding(SPLADE_MODEL)
        self._colbert = LateInteractionTextEmbedding(COLBERT_MODEL)

    def embed_document(self, text: str) -> dict:
        """Return all three vector types for a single document block."""
        dense_vec   = next(self._dense.embed([text]))         # np.ndarray (384,)
        sparse_vec  = next(self._splade.embed([text]))        # SparseEmbedding
        colbert_mat = next(self._colbert.embed([text]))       # np.ndarray (N, 128)
        return {
            "dense":   dense_vec.tolist(),
            "sparse":  models.SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            ),
            "colbert": colbert_mat.tolist(),
        }

    def embed_query(self, text: str) -> dict:
        """Query-side embeddings (ColBERT uses different query encoding)."""
        dense_vec   = next(self._dense.embed([text]))
        sparse_vec  = next(self._splade.embed([text]))
        colbert_mat = next(self._colbert.query_embed([text]))  # query_embed, not embed
        return {
            "dense":   dense_vec.tolist(),
            "sparse":  models.SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            ),
            "colbert": colbert_mat.tolist(),
        }
```

### Indexing Upgrade

Replace `client.add()` with explicit `upsert()`:

```python
def index_blocks(self, blocks: list[DocumentBlock]) -> int:
    pipeline = EmbeddingPipeline()
    points = []
    for block in blocks:
        embeddings = pipeline.embed_document(block.text)
        points.append(models.PointStruct(
            id=self._point_id(block.block_id),
            vector={
                "dense":   embeddings["dense"],
                "sparse":  embeddings["sparse"],
                "colbert": embeddings["colbert"],
            },
            payload=self._payload(block),
        ))
    # Batch upsert (256 per batch to avoid memory spikes)
    for i in range(0, len(points), 256):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points[i:i+256],
            wait=True,
        )
    return len(blocks)
```

### Payload Schema Additions

Add `content_profile` (from adaptive chunker) and `financial_anchor_terms` to enable filtered search:

```python
def _payload(self, block: DocumentBlock) -> dict:
    base = {... existing fields ...}
    base["content_profile"] = block.metadata.get("content_profile", "generic")
    base["financial_anchor_terms"] = block.metadata.get("financial_anchor_terms", [])
    base["word_count"] = len(block.text.split())
    return base
```

Create indexes for new fields:
```python
self.client.create_payload_index("contextiq_v2", "content_profile", models.PayloadSchemaType.KEYWORD)
self.client.create_payload_index("contextiq_v2", "word_count", models.PayloadSchemaType.INTEGER)
```

---

## Phase 3: Hybrid Retrieval with ColBERT Reranking

### The Full 3-Stage Pipeline

```
Stage 1: Candidate Generation (Qdrant server-side)
  ├── Dense prefetch: top-50 by cosine similarity (semantic)
  └── Sparse prefetch: top-50 by SPLADE dot product (lexical + expansion)

Stage 2: Fusion (Qdrant server-side)
  └── RRF over {dense ∪ sparse} → top-20 candidates

Stage 3: ColBERT Reranking (Qdrant server-side)
  └── MaxSim late interaction on top-20 → final top-10
```

### Nested Prefetch Query (Qdrant Query API)

```python
def search_hybrid_with_reranking(
    self,
    query: str,
    limit: int = 10,
    document_id: str | None = None,
    rrf_candidates: int = 20,
    sparse_limit: int = 50,
    dense_limit: int = 50,
) -> list[VectorSearchHit]:
    embeddings = self._embedding_pipeline.embed_query(query)
    query_filter = self._document_filter(document_id)

    # Intent routing — determines search configuration
    intent_config = self._route_query_intent(query)

    response = self.client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            # Outer prefetch: RRF fusion of dense + sparse
            models.Prefetch(
                prefetch=[
                    # Dense: semantic similarity
                    models.Prefetch(
                        query=embeddings["dense"],
                        using="dense",
                        limit=dense_limit,
                        filter=query_filter,
                    ),
                    # Sparse: SPLADE neural lexical
                    models.Prefetch(
                        query=embeddings["sparse"],
                        using="sparse",
                        limit=sparse_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=rrf_candidates,
            ),
        ],
        # Inner reranking: ColBERT MaxSim on RRF candidates
        query=embeddings["colbert"],
        using="colbert",
        limit=limit,
        with_payload=True,
    )

    return [
        VectorSearchHit(
            block_id=self._block_id_from_point(p),
            score=float(p.score or 0.0),
            section_id=self._section_id_from_point(p),
        )
        for p in response.points
    ]
```

### Query Intent Routing

Different query types need different retrieval configurations. Add `QueryIntentRouter` that adjusts the prefetch limits and optional Qdrant filters:

```python
from dataclasses import dataclass
from contextiq.retrieval.query import QueryIntent, QueryAnalyzer


@dataclass
class IntentSearchConfig:
    dense_limit: int = 50
    sparse_limit: int = 50
    rrf_candidates: int = 20
    block_type_filter: str | None = None   # "table", "paragraph", etc.
    content_profile_filter: str | None = None


class QueryIntentRouter:
    """Map query intent to Qdrant search configuration."""

    def __init__(self, analyzer: QueryAnalyzer | None = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def route(self, query: str) -> IntentSearchConfig:
        analysis = self.analyzer.analyze(query)
        intent = analysis.intent

        if intent == QueryIntent.FINANCIAL_PERFORMANCE:
            # Financial queries: boost sparse (SPLADE catches ticker symbols, $amounts)
            # Prefer tables and financial_table content profile
            return IntentSearchConfig(
                dense_limit=30,
                sparse_limit=70,         # More sparse candidates
                rrf_candidates=25,
                content_profile_filter="financial_table",  # Optional table bias
            )

        if intent == QueryIntent.ANALYTICAL:
            # "Why did X happen?" → prose explanations, not tables
            return IntentSearchConfig(
                dense_limit=70,          # More dense = more semantic
                sparse_limit=30,
                rrf_candidates=20,
                content_profile_filter="narrative_para",   # Prefer explanatory prose
            )

        if intent == QueryIntent.RISK_COMPLIANCE:
            # Risk factor queries → risk sections
            return IntentSearchConfig(
                dense_limit=50,
                sparse_limit=50,
                rrf_candidates=20,
                content_profile_filter="risk_section",
            )

        # Default: balanced hybrid
        return IntentSearchConfig()
```

### Content-Profile Filter Application

```python
def _intent_filter(
    self,
    document_id: str | None,
    intent_config: IntentSearchConfig,
) -> models.Filter | None:
    conditions: list[models.Condition] = []

    if document_id:
        conditions.append(models.FieldCondition(
            key="document_id",
            match=models.MatchValue(value=document_id),
        ))

    # Soft content profile bias: prefer blocks matching intent
    # (applied at prefetch level, not as hard filter on final results)
    return models.Filter(must=conditions) if conditions else None
```

---

## Phase 4: CandidateGenerator Refactor — Remove Python Linear Scans

### The Performance Problem

The current `CandidateGenerator` runs loops like this:

```python
for block in self.blocks_provider():   # Iterates ALL 7,000+ corpus blocks
    text = " ".join([block.text, *block.section_path]).lower()
    score = sum(1 for term in terms if term in text)
```

For the full 4-document corpus (7,197 blocks), this is 7,197 × N string comparisons per query, in Python, in-process. This was fine at 100 blocks. At enterprise scale (50K blocks), this would be the bottleneck.

### Solution: Push Everything Into Qdrant

Replace Python-side linear scans with Qdrant payload-indexed lookups:

```python
class CandidateGenerator:
    """All candidate generation goes through Qdrant — no Python linear scans."""

    def generate_with_trace(self, query: str, limit: int) -> list[RetrievalCandidate]:
        analysis = self.analyzer.analyze(query)
        intent_config = self.router.route(query)

        # Structured codes → Qdrant text match on financial_anchor_terms payload
        if analysis.has_structured_codes:
            return self._qdrant_structured_code_search(analysis.structured_codes, limit)

        # All other intents → unified hybrid search with intent routing
        hits = self.vector_index.search_hybrid_with_reranking(
            query=query,
            limit=limit,
            intent_config=intent_config,
        )
        return [RetrievalCandidate(block=self._load_block(h.block_id), stages=["hybrid_reranked"])
                for h in hits]

    def _qdrant_structured_code_search(self, codes: list[str], limit: int):
        """Use Qdrant full-text search on indexed payload for structured codes like CN-P-103."""
        response = self.vector_index.client.scroll(
            collection_name=self.vector_index.collection_name,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="block_text_searchable",
                        match=models.MatchText(text=code),
                    )
                    for code in codes
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [RetrievalCandidate(block=self._load_block(p.payload["block_id"]), stages=["structured_code"])
                for p, _ in response]
```

For full-text match on block text, we add a `full_text` payload index:
```python
self.client.create_payload_index(
    collection_name="contextiq_v2",
    field_name="block_text_searchable",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=30,
        lowercase=True,
    ),
)
```

---

## Phase 5: Quantization for Production

**Scalar quantization** (INT8) is already in the collection schema (Phase 2). Benefits:
- 4x memory reduction on dense + ColBERT vectors
- Quality impact: ~0.5% NDCG@10 reduction (from Qdrant's own benchmarks)
- Always-RAM quantized vectors: fast retrieval without disk I/O

**Oversampling** to compensate for quantization precision loss:
```python
models.SearchParams(
    quantization=models.QuantizationSearchParams(
        ignore=False,
        rescore=True,     # Re-score with full-precision after quantized retrieval
        oversampling=2.5, # Retrieve 2.5x candidates, rescore, return top-k
    )
)
```

---

## Implementation Roadmap

### Sprint 1: SPLADE + Schema Upgrade (Week 1)
1. Create `contextiq/ingestion/embedding_pipeline.py` with `EmbeddingPipeline`
2. Upgrade `VectorIndex` to use explicit `upsert()` with 3-vector points
3. Write migration script: re-index existing corpus into new `contextiq_v2` collection
4. Update eval harness to test SPLADE recall — target: recall@10 ≥ 0.60

Effort: ~2 days. Entirely backwards compatible (new collection name).

### Sprint 2: ColBERT Reranking (Week 1-2)
1. Add `search_hybrid_with_reranking()` to `VectorIndex`
2. Add `QueryIntentRouter` and `IntentSearchConfig`
3. Wire into `CandidateGenerator.generate_with_trace()`
4. Run eval: target recall@10 ≥ 0.68 (ColBERT reranking over SPLADE)

Effort: ~1.5 days. Most of the work is in the Qdrant query construction.

### Sprint 3: Adaptive Chunking (Week 2)
1. Create `contextiq/ingestion/adaptive_chunker.py` with `AdaptiveChunker`
2. Wire into `BatchIngestor.ingest()` after Docling parsing
3. Add `content_profile` to payload schema and Qdrant index
4. Re-ingest Apple 10-K with adaptive chunking
5. Run eval: target recall@10 ≥ 0.75 (better chunks = better retrieval)

Effort: ~2 days. The longest sprint because it requires corpus re-ingestion.

### Sprint 4: CandidateGenerator Refactor + Payload Text Index (Week 2-3)
1. Add `block_text_searchable` payload field + full-text index
2. Refactor Python linear scans → Qdrant `scroll()` + text match
3. Benchmark: confirm query latency < 500ms on full 7K-block corpus
4. Update all eval tests to use new search path

Effort: ~1 day. Mostly refactoring.

---

## Expected Recall Improvements

| Phase | Change | Expected Recall@10 |
|-------|--------|-------------------|
| Current baseline | BM25 + dense + RRF | 0.48 |
| + SPLADE neural sparse | Term expansion for financial/technical queries | 0.60 |
| + ColBERT reranking | 3-stage pipeline: dense+SPLADE→RRF→ColBERT | 0.68 |
| + Adaptive chunking | Right-sized blocks per content type | 0.75+ |
| + Intent routing | Prose bias for Why/How queries | 0.78+ |

---

## Dependencies to Add

```toml
# pyproject.toml additions
[project.dependencies]
fastembed = ">=0.4.0"           # Already present (BAAI/bge-small-en)
# New models auto-downloaded by fastembed:
# - prithivida/Splade_PP_en_v1 (532 MB) 
# - colbert-ir/colbertv2.0 (440 MB)
# No new Python packages needed — fastembed handles all model downloads
```

**Disk**: ~1 GB additional for SPLADE + ColBERT models (one-time download)
**RAM**: +~400 MB during indexing (ColBERT generates N×128 tensors per block)
**Indexing time**: ~3x longer (3 models vs 1), acceptable for batch ingestion
**Query time**: Dense prefetch 50 + sparse prefetch 50 + ColBERT rerank 20 = same Qdrant round trip (server-side pipeline)

---

## Eval Gates (Go / No-Go per Sprint)

```python
# tests/evals/thresholds.py
RECALL_THRESHOLDS = {
    "apple_bm25_baseline":    0.35,   # Current: lexical fallback
    "apple_vector_baseline":  0.45,   # Current: dense + BM25
    "apple_splade":           0.58,   # Sprint 1 target
    "apple_splade_colbert":   0.66,   # Sprint 2 target
    "apple_adaptive_full":    0.73,   # Sprint 3 target
    "nasa_lunar":             0.70,   # NASA corpus target
}
```

Every sprint includes a re-run of `test_seed_qrels_recall_against_apple_corpus` with the updated threshold. This is the build-time contract.

---

## Architecture Diagram (Post-Upgrade)

```
Query: "Why did Apple Services revenue increase in 2025?"

QueryIntentRouter
└── intent=ANALYTICAL → dense_limit=70, sparse_limit=30, content_profile=narrative_para

EmbeddingPipeline.embed_query()
├── dense_vec   (384 floats, BAAI/bge-small-en)
├── sparse_vec  (SparseVector, SPLADE prithivida)
└── colbert_mat (N×128 floats, colbert-ir/colbertv2.0)

Qdrant Query API (single network round-trip)
└── query_points(
      prefetch=[
        Prefetch(                           ← Stage 1: RRF fusion
          prefetch=[
            Prefetch(dense_vec, limit=70),  ← Semantic candidates
            Prefetch(sparse_vec, limit=30), ← Lexical candidates
          ],
          query=FusionQuery(RRF),
          limit=20,
        )
      ],
      query=colbert_mat,                    ← Stage 2: MaxSim reranking
      using="colbert",
      limit=10,
    )

ParentResolver
└── Enrich top-10 with parent section text (max 1500 words)

ContextEngine
└── Token budget enforcement → ContextPacket

LLM Answerer (Claude Sonnet)
└── Answer with citations
```

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `src/contextiq/ingestion/adaptive_chunker.py` | CREATE | ContentProfile classifier + chunk strategies |
| `src/contextiq/ingestion/embedding_pipeline.py` | CREATE | Dense + SPLADE + ColBERT unified pipeline |
| `src/contextiq/retrieval/vector_index.py` | MODIFY | Add 3-vector schema, `search_hybrid_with_reranking()` |
| `src/contextiq/retrieval/intent_router.py` | CREATE | QueryIntentRouter + IntentSearchConfig |
| `src/contextiq/retrieval/candidates.py` | MODIFY | Remove Python linear scans, use Qdrant |
| `scripts/reindex_corpus.py` | CREATE | Migration script for existing corpus |
| `tests/evals/thresholds.py` | CREATE | Go/no-go recall gate constants |
| `tests/unit/test_integration_e2e.py` | MODIFY | Update thresholds per sprint |

---

*Plan authored: 2026-05-27 | Research: Qdrant docs (hybrid-queries, sparse-vectors, late-interaction-models, fastembed-colbert, fastembed-splade, quantization) + arxiv:2603.25333*
