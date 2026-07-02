"""Qdrant/FastEmbed-backed hybrid retrieval index.

Dense (BAAI/bge-small-en) + BM25 sparse over one Qdrant collection, fused with
reciprocal-rank fusion. Falls back to an in-memory keyword scan when FastEmbed
models are unavailable.
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

DENSE_MODEL = "BAAI/bge-small-en"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "fast-bge-small-en"
SPARSE_VECTOR_NAME = "fast-sparse-bm25"


@dataclass(frozen=True)
class VectorSearchHit:
    """One vector search result with score and payload."""

    block_id: str
    score: float
    section_id: str | None = None


class VectorIndex:
    """Qdrant wrapper: hybrid dense + BM25 retrieval with an in-memory fallback."""

    def __init__(
        self,
        path: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.path = path or settings.qdrant_path
        self.collection_name = collection_name or settings.qdrant_collection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self._models_ready = False
        self._payload_indexes_ready = False
        self._fallback_blocks: list[DocumentBlock] = []

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        """Index document blocks into the Qdrant collection."""
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

    def search_hybrid(
        self,
        query: str,
        limit: int = 8,
        document_id: str | None = None,
        *,
        group_by_section: bool = False,
        group_size: int = 2,
    ) -> list[VectorSearchHit]:
        """Hybrid dense + BM25 sparse search with RRF fusion."""
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

    # ── Helpers ───────────────────────────────────────────────────────────────

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
        return models.NearestQuery(nearest=models.Document(text=query, model=DENSE_MODEL))

    def _sparse_query(self, query: str) -> models.NearestQuery:
        return models.NearestQuery(nearest=models.Document(text=query, model=SPARSE_MODEL))

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
            "parent_id": models.PayloadSchemaType.KEYWORD,
            "section_id": models.PayloadSchemaType.KEYWORD,
            "block_id": models.PayloadSchemaType.KEYWORD,
            "block_type": models.PayloadSchemaType.KEYWORD,
            "chunk_level": models.PayloadSchemaType.KEYWORD,
            "page": models.PayloadSchemaType.INTEGER,
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
                        "Payload index creation skipped for %s", field_name, exc_info=exc
                    )
        self._payload_indexes_ready = True

    def _delete_existing_documents(self, document_ids: set[str]) -> None:
        for document_id in document_ids:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
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
