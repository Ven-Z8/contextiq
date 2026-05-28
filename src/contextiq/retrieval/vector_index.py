"""Qdrant/FastEmbed-backed retrieval index.

Two operating modes:
  Legacy mode  — collection "contextiq_chunks_v2"
                 dense (BAAI/bge-small-en) + BM25 sparse
                 Used for backward-compat with existing indexed corpus.

  Enterprise mode — collection "contextiq_enterprise"
                    dense + SPLADE neural sparse + ColBERT late-interaction
                    Hybrid pipeline: dense+SPLADE → RRF fusion → ColBERT reranking
                    Target: recall@10 ≥ 0.75 on enterprise document corpora.

The enterprise collection is created explicitly via `create_enterprise_collection()`.
The legacy collection is still created implicitly via `client.add()`.
Both share the same Qdrant local path (separate collection names).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from contextiq.core.config import get_settings
from contextiq.ingestion.models import DocumentBlock

logger = logging.getLogger(__name__)

# ── Legacy collection (backward compat) ──────────────────────────────────────
DENSE_MODEL = "BAAI/bge-small-en"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "fast-bge-small-en"
SPARSE_VECTOR_NAME = "fast-sparse-bm25"

# ── Enterprise collection (3-vector pipeline) ─────────────────────────────────
ENTERPRISE_DENSE_NAME = "dense"
ENTERPRISE_SPARSE_NAME = "sparse_splade"
ENTERPRISE_COLBERT_NAME = "colbert"
ENTERPRISE_COLLECTION_SUFFIX = "_enterprise"

DENSE_DIM = 384    # BAAI/bge-small-en output dim
COLBERT_DIM = 128  # colbert-ir/colbertv2.0 output dim per token


@dataclass(frozen=True)
class VectorSearchHit:
    """One vector search result with score and payload."""

    block_id: str
    score: float
    section_id: str | None = None
    payload: dict | None = None  # Full Qdrant payload — used to reconstruct adaptive sub-chunks


class VectorIndex:
    """Qdrant wrapper supporting both legacy and enterprise retrieval pipelines.

    Args:
        path: Local Qdrant storage path.
        collection_name: Base collection name. Enterprise collection appends "_enterprise".
        enable_enterprise: If True, enterprise methods are available.
                           Requires EmbeddingPipeline models to be downloadable.
    """

    def __init__(
        self,
        path: Path | None = None,
        collection_name: str | None = None,
        enable_enterprise: bool = True,
    ) -> None:
        settings = get_settings()
        self.path = path or settings.qdrant_path
        self.collection_name = collection_name or settings.qdrant_collection
        self.enterprise_collection_name = self.collection_name + ENTERPRISE_COLLECTION_SUFFIX
        self.enable_enterprise = enable_enterprise
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self._models_ready = False
        self._payload_indexes_ready = False
        self._enterprise_ready = False
        self._enterprise_pipeline = None
        self._intent_router = None
        self._fallback_blocks: list[DocumentBlock] = []

    # ── Enterprise Collection ─────────────────────────────────────────────────

    @property
    def _pipeline(self):
        """Lazy-load the enterprise EmbeddingPipeline."""
        if self._enterprise_pipeline is None:
            from contextiq.ingestion.embedding_pipeline import EmbeddingPipeline
            self._enterprise_pipeline = EmbeddingPipeline(use_colbert=True)
        return self._enterprise_pipeline

    @property
    def _router(self):
        """Lazy-load the QueryIntentRouter."""
        if self._intent_router is None:
            from contextiq.retrieval.intent_router import QueryIntentRouter
            self._intent_router = QueryIntentRouter()
        return self._intent_router

    def create_enterprise_collection(self) -> bool:
        """Create the enterprise 3-vector collection if it doesn't exist.

        Returns True if created or already exists, False on error.
        """
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.enterprise_collection_name in existing:
                logger.info("Enterprise collection already exists: %s", self.enterprise_collection_name)
                return True

            self.client.create_collection(
                collection_name=self.enterprise_collection_name,
                vectors_config={
                    # Dense: semantic retrieval (BAAI/bge-small-en, 384-dim)
                    ENTERPRISE_DENSE_NAME: models.VectorParams(
                        size=DENSE_DIM,
                        distance=models.Distance.COSINE,
                        on_disk=False,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                    ),
                    # ColBERT: late-interaction reranking (128-dim per token, MaxSim)
                    # HNSW disabled (m=0) — reranking only, not first-stage retrieval
                    ENTERPRISE_COLBERT_NAME: models.VectorParams(
                        size=COLBERT_DIM,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                    ),
                },
                sparse_vectors_config={
                    # SPLADE: neural sparse with term expansion
                    ENTERPRISE_SPARSE_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    ),
                },
                # Scalar quantization: 4x memory reduction, <0.5% quality impact
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )

            # Payload indexes for filtered search
            payload_indexes = {
                "document_id":       models.PayloadSchemaType.KEYWORD,
                "block_id":          models.PayloadSchemaType.KEYWORD,
                "section_id":        models.PayloadSchemaType.KEYWORD,
                "parent_id":         models.PayloadSchemaType.KEYWORD,
                "block_type":        models.PayloadSchemaType.KEYWORD,
                "chunk_level":       models.PayloadSchemaType.KEYWORD,
                "content_profile":   models.PayloadSchemaType.KEYWORD,
                "page":              models.PayloadSchemaType.INTEGER,
                "word_count":        models.PayloadSchemaType.INTEGER,
            }
            for field_name, schema in payload_indexes.items():
                self.client.create_payload_index(
                    collection_name=self.enterprise_collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )

            # Full-text index on block text for structured code exact matching
            self.client.create_payload_index(
                collection_name=self.enterprise_collection_name,
                field_name="block_text_searchable",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                ),
            )

            logger.info("Enterprise collection created: %s", self.enterprise_collection_name)
            self._enterprise_ready = True
            return True

        except Exception as exc:
            logger.warning("Failed to create enterprise collection", exc_info=exc)
            return False

    def index_blocks_enterprise(self, blocks: list[DocumentBlock]) -> int:
        """Index blocks into the enterprise 3-vector collection.

        Generates dense + SPLADE + ColBERT embeddings and upserts all
        into the enterprise collection. Deletes existing points for the
        same document_id first (idempotent re-ingestion).

        Args:
            blocks: Document blocks to index.

        Returns:
            Number of blocks indexed.
        """
        if not blocks:
            return 0

        if not self.create_enterprise_collection():
            logger.error("Cannot index — enterprise collection creation failed")
            return 0

        # Delete existing points for these documents
        document_ids = {block.document_id for block in blocks}
        self._delete_existing_documents_from(document_ids, self.enterprise_collection_name)

        # Batch embed all blocks
        texts = [block.text for block in blocks]
        try:
            embeddings = self._pipeline.embed_documents_batch(texts, batch_size=16)
        except Exception as exc:
            logger.error("Embedding pipeline failed during indexing", exc_info=exc)
            return 0

        # Build Qdrant points
        points: list[models.PointStruct] = []
        for block, emb in zip(blocks, embeddings):
            vector_dict: dict = {
                ENTERPRISE_DENSE_NAME: emb.dense,
                ENTERPRISE_SPARSE_NAME: models.SparseVector(
                    indices=emb.sparse_indices,
                    values=emb.sparse_values,
                ),
            }
            if emb.colbert is not None:
                vector_dict[ENTERPRISE_COLBERT_NAME] = emb.colbert

            points.append(models.PointStruct(
                id=self._point_id(block.block_id),
                vector=vector_dict,
                payload=self._enterprise_payload(block),
            ))

        # Upsert in batches of 64 (ColBERT multivectors are large payloads)
        batch_size = 64
        indexed = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.enterprise_collection_name,
                    points=batch,
                    wait=True,
                )
                indexed += len(batch)
                logger.debug(
                    "Enterprise indexed %d/%d blocks", indexed, len(points)
                )
            except Exception as exc:
                logger.error("Upsert failed at batch %d", i, exc_info=exc)
                raise

        logger.info(
            "Enterprise collection: %d blocks indexed for %d documents",
            indexed, len(document_ids),
        )
        return indexed

    def search_hybrid_with_reranking(
        self,
        query: str,
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchHit]:
        """3-stage enterprise retrieval pipeline with ColBERT reranking.

        Pipeline:
            Stage 1a: Dense semantic prefetch (BAAI/bge-small-en)
            Stage 1b: SPLADE neural sparse prefetch (term expansion)
            Stage 2:  RRF fusion of stage 1 candidates
            Stage 3:  ColBERT late-interaction reranking (MaxSim)

        Intent routing automatically adjusts dense/sparse prefetch limits
        based on query type (financial → sparse-heavy; analytical → dense-heavy).
        """
        if not self._enterprise_collection_exists():
            logger.warning(
                "Enterprise collection not found — call index_blocks_enterprise() first"
            )
            return []

        try:
            embeddings = self._pipeline.embed_query(query)
        except Exception as exc:
            logger.warning("Enterprise embedding failed; no results", exc_info=exc)
            return []

        config = self._router.route(query)
        logger.debug("Intent routing: %s", config.description)

        query_filter = self._document_filter(document_id)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")

                prefetch_args = dict(filter=query_filter) if query_filter else {}

                response = self.client.query_points(
                    collection_name=self.enterprise_collection_name,
                    prefetch=[
                        # Outer prefetch: RRF fusion of dense + sparse
                        models.Prefetch(
                            prefetch=[
                                # Stage 1a: Dense semantic retrieval
                                models.Prefetch(
                                    query=embeddings.dense,
                                    using=ENTERPRISE_DENSE_NAME,
                                    limit=config.dense_limit,
                                    **prefetch_args,
                                ),
                                # Stage 1b: SPLADE neural sparse retrieval
                                models.Prefetch(
                                    query=models.SparseVector(
                                        indices=embeddings.sparse_indices,
                                        values=embeddings.sparse_values,
                                    ),
                                    using=ENTERPRISE_SPARSE_NAME,
                                    limit=config.sparse_limit,
                                    **prefetch_args,
                                ),
                            ],
                            # Stage 2: RRF fusion
                            query=models.FusionQuery(fusion=models.Fusion.RRF),
                            limit=config.rrf_candidates,
                        ),
                    ],
                    # Stage 3: ColBERT late-interaction reranking
                    query=embeddings.colbert,
                    using=ENTERPRISE_COLBERT_NAME,
                    limit=limit,
                    with_payload=True,
                )

        except Exception as exc:
            if self._is_missing_collection_error(exc):
                return []
            logger.warning("Enterprise search failed", exc_info=exc)
            return []

        return [
            VectorSearchHit(
                block_id=self._block_id_from_point(p),
                score=float(p.score or 0.0),
                section_id=self._section_id_from_point(p),
                payload=p.payload or {},
            )
            for p in response.points
        ]

    def search_enterprise(
        self,
        query: str,
        limit: int = 8,
        document_id: str | None = None,
    ) -> list[str]:
        """Return block IDs from the enterprise pipeline."""
        return [h.block_id for h in self.search_hybrid_with_reranking(
            query=query,
            limit=limit,
            document_id=document_id,
        )]

    def enterprise_collection_info(self) -> dict | None:
        """Return collection stats for monitoring (None if not created)."""
        if not self._enterprise_collection_exists():
            return None
        try:
            info = self.client.get_collection(self.enterprise_collection_name)
        except Exception as exc:
            logger.warning("Could not get enterprise collection info", exc_info=exc)
            return None

        # vectors_count was deprecated in qdrant-client ≥ 1.9; fall back to count()
        points_count = getattr(info, "points_count", None)
        if points_count is None:
            try:
                points_count = self.client.count(
                    self.enterprise_collection_name, exact=True
                ).count
            except Exception:
                points_count = -1

        vectors_count = getattr(info, "vectors_count", None)

        return {
            "name": self.enterprise_collection_name,
            "points_count": points_count,
            "vectors_count": vectors_count,
            "status": str(getattr(info, "status", "unknown")),
        }

    # ── Legacy Collection (backward compat) ───────────────────────────────────

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        """Index document blocks into the legacy collection (backward compat)."""
        if not blocks:
            return 0

        if not self._ensure_models():
            self._index_fallback(blocks)
            return len(blocks)

        self._delete_existing_documents({block.document_id for block in blocks})
        self._ensure_payload_indexes()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
            self.client.add(
                collection_name=self.collection_name,
                documents=[block.text for block in blocks],
                metadata=[self._payload(block) for block in blocks],
                ids=[self._point_id(block.block_id) for block in blocks],
            )
        return len(blocks)

    def search(
        self,
        query: str,
        limit: int = 8,
        document_id: str | None = None,
        *,
        group_by_section: bool = False,
    ) -> list[str]:
        """Return matching block ids (legacy pipeline)."""
        return [
            hit.block_id
            for hit in self.search_hybrid(
                query=query,
                limit=limit,
                document_id=document_id,
                group_by_section=group_by_section,
            )
        ]

    def search_sparse(
        self,
        query: str,
        limit: int = 8,
        document_id: str | None = None,
    ) -> list[str]:
        """Return sparse/BM25 block ids (legacy pipeline)."""
        if self._fallback_blocks:
            return self._search_fallback(query=query, limit=limit, document_id=document_id)
        if not self._ensure_models():
            return []

        query_filter = self._document_filter(document_id)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=self._sparse_query(query),
                            using=SPARSE_VECTOR_NAME,
                            limit=max(limit * 4, 24),
                            filter=query_filter,
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    query_filter=query_filter,
                )
        except Exception as exc:
            if self._is_missing_collection_error(exc):
                return []
            raise
        return [self._block_id_from_point(point) for point in response.points]

    def search_hybrid(
        self,
        query: str,
        limit: int = 8,
        document_id: str | None = None,
        *,
        group_by_section: bool = False,
        group_size: int = 2,
    ) -> list[VectorSearchHit]:
        """Legacy hybrid dense + BM25 sparse search with RRF fusion."""
        if self._fallback_blocks:
            return [
                VectorSearchHit(block_id=block_id, score=float(limit - index))
                for index, block_id in enumerate(
                    self._search_fallback(query=query, limit=limit, document_id=document_id)
                )
            ]
        if not self._ensure_models():
            return []

        query_filter = self._document_filter(document_id)
        prefetch = [
            models.Prefetch(
                query=self._dense_query(query),
                using=DENSE_VECTOR_NAME,
                limit=max(limit * 8, 40),
                filter=query_filter,
            ),
            models.Prefetch(
                query=self._sparse_query(query),
                using=SPARSE_VECTOR_NAME,
                limit=max(limit * 8, 40),
                filter=query_filter,
            ),
        ]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
                if group_by_section:
                    response = self.client.query_points_groups(
                        collection_name=self.collection_name,
                        group_by="section_id",
                        prefetch=prefetch,
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=limit,
                        group_size=group_size,
                        query_filter=query_filter,
                    )
                    hits: list[VectorSearchHit] = []
                    for group in response.groups:
                        for point in group.hits:
                            hits.append(
                                VectorSearchHit(
                                    block_id=self._block_id_from_point(point),
                                    score=float(point.score or 0.0),
                                    section_id=str(group.id),
                                )
                            )
                    return hits

                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    query_filter=query_filter,
                )
        except Exception as exc:
            if self._is_missing_collection_error(exc):
                return []
            raise

        return [
            VectorSearchHit(
                block_id=self._block_id_from_point(point),
                score=float(point.score or 0.0),
                section_id=self._section_id_from_point(point),
            )
            for point in response.points
        ]

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _enterprise_payload(self, block: DocumentBlock) -> dict:
        """Payload for enterprise collection — richer schema than legacy."""
        base = self._payload(block)
        base["content_profile"] = block.metadata.get("content_profile", "generic")
        base["word_count"] = len(block.text.split())
        base["block_text_searchable"] = block.text  # for full-text index
        return base

    def _payload(self, block: DocumentBlock) -> dict:
        section_id = block.metadata.get("section_id")
        parent_id = block.metadata.get("parent_id")
        chunk_level = block.metadata.get("chunk_level", "child")
        section_id_value = (
            str(section_id) if section_id is not None else self._fallback_section_id(block)
        )
        parent_id_value = (
            str(parent_id) if parent_id is not None else self._fallback_section_id(block)
        )
        return {
            "document_id": block.document_id,
            "block_id": block.block_id,
            "source_path": block.source_path,
            "page": block.page,
            "section_path": " > ".join(block.section_path),
            "section_id": section_id_value,
            "parent_id": parent_id_value,
            "block_type": block.block_type.value,
            "chunk_level": str(chunk_level),
        }

    def _fallback_section_id(self, block: DocumentBlock) -> str:
        slug = "-".join(part.lower().replace(" ", "-") for part in block.section_path[:2])
        if not slug:
            slug = "document"
        return f"{block.document_id}:{slug}"

    def _document_filter(self, document_id: str | None) -> models.Filter | None:
        if document_id is None:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    def _dense_query(self, query: str) -> models.NearestQuery:
        return models.NearestQuery(
            nearest=models.Document(text=query, model=DENSE_MODEL),
        )

    def _sparse_query(self, query: str) -> models.NearestQuery:
        return models.NearestQuery(
            nearest=models.Document(text=query, model=SPARSE_MODEL),
        )

    def _point_id(self, block_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"contextiq:{block_id}"))

    def _block_id_from_point(self, point: models.ScoredPoint) -> str:
        payload = point.payload or {}
        block_id = payload.get("block_id")
        if isinstance(block_id, str):
            return block_id
        raise ValueError("Vector search point missing block_id payload")

    def _section_id_from_point(self, point: models.ScoredPoint) -> str | None:
        payload = point.payload or {}
        section_id = payload.get("section_id")
        return str(section_id) if section_id is not None else None

    def _enterprise_collection_exists(self) -> bool:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            return self.enterprise_collection_name in existing
        except Exception:
            return False

    def _ensure_models(self) -> bool:
        if self._models_ready:
            return True
        try:
            self.client.set_model(DENSE_MODEL)
            self.client.set_sparse_model(SPARSE_MODEL)
        except Exception as exc:
            logger.warning(
                "FastEmbed models unavailable; using lexical vector-index fallback",
                exc_info=exc,
            )
            return False
        self._models_ready = True
        return True

    def _ensure_payload_indexes(self) -> None:
        if self._payload_indexes_ready:
            return

        index_fields = {
            "document_id": models.PayloadSchemaType.KEYWORD,
            "parent_id":   models.PayloadSchemaType.KEYWORD,
            "section_id":  models.PayloadSchemaType.KEYWORD,
            "block_id":    models.PayloadSchemaType.KEYWORD,
            "block_type":  models.PayloadSchemaType.KEYWORD,
            "chunk_level": models.PayloadSchemaType.KEYWORD,
            "page":        models.PayloadSchemaType.INTEGER,
        }
        for field_name, schema in index_fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                if not self._is_missing_collection_error(exc):
                    logger.debug(
                        "Payload index creation skipped for %s",
                        field_name,
                        exc_info=exc,
                    )
        self._payload_indexes_ready = True

    def _delete_existing_documents(self, document_ids: set[str]) -> None:
        self._delete_existing_documents_from(document_ids, self.collection_name)

    def _delete_existing_documents_from(
        self, document_ids: set[str], collection_name: str
    ) -> None:
        for document_id in document_ids:
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    ),
                    wait=True,
                )
            except Exception as exc:
                if self._is_missing_collection_error(exc):
                    continue
                raise RuntimeError(
                    f"Failed to delete vectors for document {document_id}"
                ) from exc

    def _index_fallback(self, blocks: list[DocumentBlock]) -> None:
        document_ids = {block.document_id for block in blocks}
        self._fallback_blocks = [
            block for block in self._fallback_blocks if block.document_id not in document_ids
        ]
        self._fallback_blocks.extend(blocks)

    def _search_fallback(
        self,
        query: str,
        limit: int,
        document_id: str | None = None,
    ) -> list[str]:
        terms = {
            term.strip(".,:;!?()[]{}\"'").lower()
            for term in query.split()
            if len(term.strip(".,:;!?()[]{}\"'")) > 2
        }
        scored: list[tuple[int, str]] = []
        for block in self._fallback_blocks:
            if document_id is not None and block.document_id != document_id:
                continue
            text = " ".join([block.text, *block.section_path]).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, block.block_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block_id for _, block_id in scored[:limit]]

    def _is_missing_collection_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "not found" in message
            or "doesn't exist" in message
            or "does not exist" in message
            or ("collection" in message and "not found" in message)
        )
