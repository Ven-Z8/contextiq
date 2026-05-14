"""Qdrant/FastEmbed-backed retrieval index."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from contextiq.core.config import get_settings
from contextiq.ingestion.models import DocumentBlock

logger = logging.getLogger(__name__)


class VectorIndex:
    """Small Qdrant wrapper for local semantic retrieval."""

    def __init__(self, path: Path | None = None, collection_name: str = "contextiq_blocks") -> None:
        settings = get_settings()
        self.path = path or settings.qdrant_path
        self.collection_name = collection_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self._model_ready = False
        self._fallback_blocks: list[DocumentBlock] = []

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        """Index document blocks and return the number indexed."""

        if not blocks:
            return 0

        if not self._ensure_model():
            self._index_fallback(blocks)
            return len(blocks)

        self._delete_existing_documents({block.document_id for block in blocks})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
            self.client.add(
                collection_name=self.collection_name,
                documents=[block.text for block in blocks],
                metadata=[
                    {
                        "document_id": block.document_id,
                        "block_id": block.block_id,
                        "source_path": block.source_path,
                        "page": block.page,
                        "section_path": " > ".join(block.section_path),
                        "block_type": block.block_type.value,
                    }
                    for block in blocks
                ],
                ids=[self._point_id(block.block_id) for block in blocks],
            )
        return len(blocks)

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
                    logger.info(
                        "Skipping vector delete because collection does not exist yet",
                        exc_info=exc,
                    )
                    continue
                raise RuntimeError(
                    f"Failed to delete existing vectors for document {document_id}"
                ) from exc

    def search(self, query: str, limit: int = 8) -> list[str]:
        """Return matching block ids for a query."""

        if self._fallback_blocks:
            return self._search_fallback(query=query, limit=limit)
        if not self._ensure_model():
            return []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
            responses = self.client.query(
                collection_name=self.collection_name,
                query_text=query,
                limit=limit,
            )
        return [str(response.metadata["block_id"]) for response in responses]

    def _point_id(self, block_id: str) -> str:
        """Return a Qdrant-compatible deterministic id for a citation block id."""

        return str(uuid5(NAMESPACE_URL, f"contextiq:{block_id}"))

    def _ensure_model(self) -> bool:
        if self._model_ready:
            return True
        try:
            self.client.set_model("BAAI/bge-small-en")
        except Exception as exc:
            logger.warning(
                "FastEmbed model unavailable; using lexical vector-index fallback",
                exc_info=exc,
            )
            return False
        self._model_ready = True
        return True

    def _index_fallback(self, blocks: list[DocumentBlock]) -> None:
        document_ids = {block.document_id for block in blocks}
        self._fallback_blocks = [
            block for block in self._fallback_blocks if block.document_id not in document_ids
        ]
        self._fallback_blocks.extend(blocks)

    def _search_fallback(self, query: str, limit: int) -> list[str]:
        terms = {
            term.strip(".,:;!?()[]{}\"'").lower()
            for term in query.split()
            if len(term.strip(".,:;!?()[]{}\"'")) > 2
        }
        scored: list[tuple[int, str]] = []
        for block in self._fallback_blocks:
            text = " ".join([block.text, *block.section_path]).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, block.block_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block_id for _, block_id in scored[:limit]]

    def _is_missing_collection_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "not found" in message or "doesn't exist" in message or "does not exist" in message
