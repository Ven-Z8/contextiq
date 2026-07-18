"""Qdrant/FastEmbed-backed hybrid retrieval index.

Dense (BAAI/bge-small-en) + BM25 sparse over one Qdrant collection, fused with
reciprocal-rank fusion. Falls back to an in-memory keyword scan when FastEmbed
models are unavailable.

Optional: NVIDIA NIM hosted embeddings (llama-nemotron-embed-1b-v2) and reranker
(llama-nemotron-rerank-1b-v2) when NVIDIA_API_KEY is set.

For local/dev use, an InMemoryIndex (NumPy cosine + rank_bm25) avoids Qdrant's
file-locking limitations. Select via Settings.vector_backend ('qdrant' | 'memory').
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
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


class NvidiaEmbeddingClient:
    """NVIDIA NIM hosted embedding client (OpenAI-compatible API)."""

    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = __import__("httpx").Client(timeout=60.0)

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        """Embed a list of texts. input_type: 'query' or 'passage'."""
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        model = "nvidia/llama-nemotron-embed-1b-v2"
        payload = {"model": model, "input": texts, "input_type": input_type}
        resp = self.client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


class NvidiaRerankClient:
    """NVIDIA NIM hosted reranker client."""

    def __init__(self, api_key: str, base_url: str = "https://ai.api.nvidia.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = __import__("httpx").Client(timeout=60.0)

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Rerank passages for a query. Returns list of (index, logit) sorted by score desc."""
        url = f"{self.base_url}/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "nvidia/llama-nemotron-rerank-1b-v2",
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
        }
        resp = self.client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # rankings: list of {index, logit} already sorted by score desc
        return [(r["index"], r["logit"]) for r in data["rankings"]]


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

        # NVIDIA NIM clients (lazy init)
        self._nvidia_embed_client: NvidiaEmbeddingClient | None = None
        self._nvidia_rerank_client: NvidiaRerankClient | None = None
        self._nvidia_available: bool | None = None

    def _get_nvidia_embed_client(self) -> NvidiaEmbeddingClient | None:
        if self._nvidia_embed_client is not None:
            return self._nvidia_embed_client
        settings = get_settings()
        if settings.nvidia_api_key is None:
            return None
        try:
            self._nvidia_embed_client = NvidiaEmbeddingClient(
                api_key=settings.nvidia_api_key.get_secret_value(),
                base_url=settings.nvidia_base_url,
            )
            return self._nvidia_embed_client
        except Exception as exc:
            logger.warning("Failed to init NVIDIA embedding client", exc_info=exc)
            return None

    def _get_nvidia_rerank_client(self) -> NvidiaRerankClient | None:
        if self._nvidia_rerank_client is not None:
            return self._nvidia_rerank_client
        settings = get_settings()
        if settings.nvidia_api_key is None:
            return None
        try:
            self._nvidia_rerank_client = NvidiaRerankClient(
                api_key=settings.nvidia_api_key.get_secret_value(),
            )
            return self._nvidia_rerank_client
        except Exception as exc:
            logger.warning("Failed to init NVIDIA rerank client", exc_info=exc)
            return None

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        """Index document blocks into the Qdrant collection."""
        if not blocks:
            return 0

        # Try NVIDIA embeddings first, fall back to FastEmbed
        nvidia_embed = self._get_nvidia_embed_client()
        if nvidia_embed is not None:
            return self._index_with_nvidia(blocks, nvidia_embed)

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

    def _index_with_nvidia(
        self, blocks: list[DocumentBlock], nvidia_embed: NvidiaEmbeddingClient
    ) -> int:
        """Index blocks using NVIDIA hosted embeddings."""
        self._delete_existing_documents({block.document_id for block in blocks})
        self._ensure_payload_indexes()

        # Ensure collection exists
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=2048, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(),
                },
            )

        # Batch embed
        texts = [block.text for block in blocks]
        embeddings = nvidia_embed.embed(texts, input_type="passage")

        # Also generate sparse vectors using fastembed BM25
        from fastembed.sparse import SparseTextEmbedding
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        sparse_embeddings = list(sparse_model.embed(texts))

        points = []
        for block, dense_emb, sparse_emb in zip(blocks, embeddings, sparse_embeddings, strict=True):
            sparse_vector = models.SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
            points.append(
                models.PointStruct(
                    id=self._point_id(block.block_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_emb,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=self._payload(block),
                )
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client.*")
            self.client.upsert(collection_name=self.collection_name, points=points)

        return len(blocks)

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

        # Try NVIDIA embedding for query
        nvidia_embed = self._get_nvidia_embed_client()
        if nvidia_embed is not None:
            return self._search_dense_nvidia(
                query, limit, document_id, group_by_section, group_size, nvidia_embed
            )

        if not self._ensure_models():
            return []

        return self._search_hybrid_fastembed(
            query, limit, document_id, group_by_section, group_size
        )

    def _search_dense_nvidia(
        self,
        query: str,
        limit: int,
        document_id: str | None,
        group_by_section: bool,
        group_size: int,
        nvidia_embed: NvidiaEmbeddingClient,
    ) -> list[VectorSearchHit]:
        """Hybrid search using NVIDIA embedding for dense + Qdrant BM25 for sparse."""
        query_embedding = nvidia_embed.embed([query], input_type="query")[0]
        query_filter = self._document_filter(document_id)

        # Prefetch dense (NVIDIA) + sparse (Qdrant BM25) then RRF fusion
        prefetch = [
            models.Prefetch(
                query=models.NearestQuery(nearest=query_embedding),
                using=DENSE_VECTOR_NAME,
                limit=max(limit * 4, 40),
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.NearestQuery(nearest=models.Document(text=query, model="Qdrant/bm25")),
                using=SPARSE_VECTOR_NAME,
                limit=max(limit * 4, 40),
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

    def _search_hybrid_fastembed(
        self,
        query: str,
        limit: int,
        document_id: str | None,
        group_by_section: bool,
        group_size: int,
    ) -> list[VectorSearchHit]:
        """Original FastEmbed-based hybrid search."""
        query_filter = self._document_filter(document_id)
        prefetch = [
            models.Prefetch(
                query=models.NearestQuery(nearest=models.Document(text=query, model=DENSE_MODEL)),
                using=DENSE_VECTOR_NAME,
                limit=max(limit * 8, 40),
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.NearestQuery(nearest=models.Document(text=query, model=SPARSE_MODEL)),
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

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchHit],
        top_k: int,
    ) -> list[VectorSearchHit]:
        """Rerank candidates using NVIDIA hosted reranker if available,
        else return as-is."""
        nvidia_rerank = self._get_nvidia_rerank_client()
        if nvidia_rerank is None or not candidates:
            return candidates[:top_k]

        # Load block texts for reranking
        from contextiq.retrieval.store import LocalDocumentStore
        store = LocalDocumentStore()
        blocks_by_id = {block.block_id: block for block in store.load_blocks()}

        passages = []
        for hit in candidates:
            block = blocks_by_id.get(hit.block_id)
            if block:
                passages.append(block.text)
            else:
                passages.append("")

        try:
            rankings = nvidia_rerank.rerank(query, passages)
            # Map back to hits
            ranked_hits = []
            for idx, logit in rankings:
                if 0 <= idx < len(candidates):
                    hit = candidates[idx]
                    ranked_hits.append(
                    VectorSearchHit(
                        block_id=hit.block_id, score=logit, section_id=hit.section_id
                    )
                )
            # Append any unranked
            ranked_ids = {r[0] for r in rankings}
            for i, hit in enumerate(candidates):
                if i not in ranked_ids:
                    ranked_hits.append(hit)
            return ranked_hits[:top_k]
        except Exception as exc:
            logger.warning("NVIDIA rerank failed, using original order", exc_info=exc)
            return candidates[:top_k]

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


# ---------------------------------------------------------------------------
# In-memory index for local/dev (no file locking, persists to .npz + pickle)
# ---------------------------------------------------------------------------

class InMemoryIndex:
    """NumPy cosine + rank_bm25 hybrid index.

    No file locks, no server, no Qdrant.  Persists embeddings and metadata
    to a single pickle file so re-indexing is not needed between runs.
    Uses NVIDIA NIM embeddings when available, else fastembed BGE-small.
    """

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or settings.data_dir / "memindex.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # persisted state
        self._block_ids: list[str] = []
        self._payloads: list[dict] = []
        self._embeddings: np.ndarray | None = None  # (N, dim) float32
        self._bm25 = None
        self._bm25_corpus: list[str] = []

        # NVIDIA clients (lazy)
        self._nvidia_embed_client: NvidiaEmbeddingClient | None = None
        self._nvidia_rerank_client: NvidiaRerankClient | None = None
        self._fastembed_model = None

        self._load()

    # -- NVIDIA clients (shared logic with VectorIndex) --

    def _get_nvidia_embed_client(self) -> NvidiaEmbeddingClient | None:
        if self._nvidia_embed_client is not None:
            return self._nvidia_embed_client
        settings = get_settings()
        if settings.nvidia_api_key is None:
            return None
        try:
            self._nvidia_embed_client = NvidiaEmbeddingClient(
                api_key=settings.nvidia_api_key.get_secret_value(),
                base_url=settings.nvidia_base_url,
            )
            return self._nvidia_embed_client
        except Exception as exc:
            logger.warning("InMemoryIndex: NVIDIA embed init failed", exc_info=exc)
            return None

    def _get_nvidia_rerank_client(self) -> NvidiaRerankClient | None:
        if self._nvidia_rerank_client is not None:
            return self._nvidia_rerank_client
        settings = get_settings()
        if settings.nvidia_api_key is None:
            return None
        try:
            self._nvidia_rerank_client = NvidiaRerankClient(
                api_key=settings.nvidia_api_key.get_secret_value(),
            )
            return self._nvidia_rerank_client
        except Exception as exc:
            logger.warning("InMemoryIndex: NVIDIA rerank init failed", exc_info=exc)
            return None

    # -- embedding --

    def _embed_batch(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        nvidia = self._get_nvidia_embed_client()
        if nvidia is not None:
            return nvidia.embed(texts, input_type=input_type)

        # fallback to fastembed
        if self._fastembed_model is None:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name=DENSE_MODEL)
        return [list(e) for e in self._fastembed_model.embed(texts)]

    # -- public API (same surface as VectorIndex) --

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        if not blocks:
            return 0

        # remove any existing blocks for these document_ids
        doc_ids = {b.document_id for b in blocks}
        keep = [
            i for i, bid in enumerate(self._block_ids)
            if self._payloads[i].get("document_id") not in doc_ids
        ]
        if keep and self._embeddings is not None and len(self._block_ids) > 0:
            self._block_ids = [self._block_ids[i] for i in keep]
            self._payloads = [self._payloads[i] for i in keep]
            self._embeddings = self._embeddings[keep]
            self._bm25_corpus = [self._bm25_corpus[i] for i in keep]
        else:
            if len(self._block_ids) > 0 and not keep:
                self._block_ids = []
                self._payloads = []
                self._embeddings = None
                self._bm25_corpus = []

        # embed new blocks
        texts = [b.text for b in blocks]
        new_embeddings = self._embed_batch(texts, input_type="passage")
        new_arr = np.array(new_embeddings, dtype=np.float32)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = new_arr
        else:
            self._embeddings = np.vstack([self._embeddings, new_arr])

        for block in blocks:
            self._block_ids.append(block.block_id)
            self._payloads.append(self._payload(block))
            self._bm25_corpus.append(block.text)

        # rebuild BM25
        self._rebuild_bm25()
        self._save()
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
        if self._embeddings is None or len(self._block_ids) == 0:
            return []

        # dense cosine search
        q_emb = np.array(self._embed_batch([query], input_type="query")[0], dtype=np.float32)
        # normalize
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self._embeddings / norms
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
        dense_scores = normalized @ q_emb  # cosine similarity

        # BM25 sparse search
        bm25_scores = np.zeros(len(self._block_ids), dtype=np.float32)
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_raw = self._bm25.get_scores(tokenized_query)
            # normalize bm25 to 0-1 range
            max_bm25 = max(bm25_raw) if len(bm25_raw) > 0 else 1.0
            if max_bm25 > 0:
                bm25_scores = np.array(bm25_raw, dtype=np.float32) / max_bm25

        # RRF fusion: dense rank + bm25 rank
        k = 15  # RRF constant (lower = more weight to top dense hits)
        dense_order = np.argsort(-dense_scores)
        bm25_order = np.argsort(-bm25_scores)
        dense_rank = np.empty(len(dense_scores), dtype=np.int32)
        bm25_rank = np.empty(len(bm25_scores), dtype=np.int32)
        dense_rank[dense_order] = np.arange(len(dense_scores))
        bm25_rank[bm25_order] = np.arange(len(bm25_scores))
        rrf_scores = 1.0 / (k + dense_rank) + 1.0 / (k + bm25_rank)

        # filter by document_id
        if document_id is not None:
            mask = np.array([
                p.get("document_id") == document_id for p in self._payloads
            ], dtype=bool)
            rrf_scores = np.where(mask, rrf_scores, -1.0)

        # top-k
        top_indices = np.argsort(-rrf_scores)[:max(limit * 3, 30)]

        hits: list[VectorSearchHit] = []
        for idx in top_indices:
            if rrf_scores[idx] < 0:
                continue
            hits.append(VectorSearchHit(
                block_id=self._block_ids[idx],
                score=float(rrf_scores[idx]),
                section_id=self._payloads[idx].get("section_id"),
            ))
            if len(hits) >= limit:
                break

        # group_by_section: dedupe by section_id, keeping top hit per section
        if group_by_section and hits:
            seen_sections: set[str] = set()
            grouped: list[VectorSearchHit] = []
            for hit in hits:
                sid = hit.section_id or hit.block_id
                if sid not in seen_sections:
                    seen_sections.add(sid)
                    grouped.append(hit)
                if len(grouped) >= limit:
                    break
            hits = grouped

        return hits

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchHit],
        top_k: int,
    ) -> list[VectorSearchHit]:
        nvidia_rerank = self._get_nvidia_rerank_client()
        if nvidia_rerank is None or not candidates:
            return candidates[:top_k]

        # load block texts
        from contextiq.retrieval.store import LocalDocumentStore
        store = LocalDocumentStore()
        blocks_by_id = {block.block_id: block for block in store.load_blocks()}

        passages = []
        for hit in candidates:
            block = blocks_by_id.get(hit.block_id)
            passages.append(block.text if block else "")

        try:
            rankings = nvidia_rerank.rerank(query, passages)
            ranked_hits = []
            for idx, logit in rankings:
                if 0 <= idx < len(candidates):
                    hit = candidates[idx]
                    ranked_hits.append(VectorSearchHit(
                        block_id=hit.block_id, score=logit, section_id=hit.section_id
                    ))
            ranked_ids = {r[0] for r in rankings}
            for i, hit in enumerate(candidates):
                if i not in ranked_ids:
                    ranked_hits.append(hit)
            return ranked_hits[:top_k]
        except Exception as exc:
            logger.warning("InMemoryIndex: NVIDIA rerank failed", exc_info=exc)
            return candidates[:top_k]

    # -- persistence --

    def _save(self) -> None:
        data = {
            "block_ids": self._block_ids,
            "payloads": self._payloads,
            "embeddings": self._embeddings,
            "bm25_corpus": self._bm25_corpus,
        }
        with open(self.path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            # Safe: this pickle file is written only by our own index_blocks()
            # method, never from external/untrusted input.
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self._block_ids = data.get("block_ids", [])
            self._payloads = data.get("payloads", [])
            self._embeddings = data.get("embeddings")
            self._bm25_corpus = data.get("bm25_corpus", [])
            if self._bm25_corpus:
                self._rebuild_bm25()
            logger.info(
                "InMemoryIndex loaded: %d blocks from %s",
                len(self._block_ids), self.path,
            )
        except Exception as exc:
            logger.warning("InMemoryIndex: failed to load %s", exc)

    def _rebuild_bm25(self) -> None:
        if not self._bm25_corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenized)
        except Exception as exc:
            logger.warning("InMemoryIndex: BM25 init failed", exc)
            self._bm25 = None

    def _payload(self, block: DocumentBlock) -> dict:
        section_id = block.metadata.get("section_id")
        parent_id = block.metadata.get("parent_id")
        chunk_level = block.metadata.get("chunk_level", "child")
        section_id_value = (
            str(section_id) if section_id is not None else f"{block.document_id}:fallback"
        )
        parent_id_value = (
            str(parent_id) if parent_id is not None else f"{block.document_id}:fallback"
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


def create_vector_index() -> VectorIndex | InMemoryIndex:
    """Factory: select backend based on Settings.vector_backend."""
    settings = get_settings()
    backend = getattr(settings, "vector_backend", "qdrant")
    if backend == "memory":
        logger.info("Using InMemoryIndex (dev/local backend)")
        return InMemoryIndex()
    return VectorIndex()
