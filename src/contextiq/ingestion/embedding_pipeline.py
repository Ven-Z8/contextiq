"""Unified embedding pipeline: dense + SPLADE neural sparse + ColBERT multivector.

Three-signal embedding for enterprise RAG retrieval:
  - Dense (BAAI/bge-small-en): semantic similarity, 384-dim
  - SPLADE (prithivida/Splade_PP_en_v1): neural sparse with term expansion
  - ColBERT (colbert-ir/colbertv2.0): late-interaction reranking, N×128-dim

Usage:
    pipeline = EmbeddingPipeline()
    doc_vecs = pipeline.embed_document("Apple total net sales were $416,161 million")
    query_vecs = pipeline.embed_query("What was Apple revenue in 2025?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-small-en"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
COLBERT_MODEL = "colbert-ir/colbertv2.0"


@dataclass
class EmbeddedVectors:
    """All three vector types for one text input."""

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    colbert: list[list[float]] | None = None  # None when use_colbert=False


class EmbeddingPipeline:
    """Lazy-loaded, batch-capable unified embedding pipeline.

    Models are downloaded on first use via FastEmbed (cached to ~/.cache/fastembed/).
    SPLADE: 532 MB download, ColBERT: 440 MB download — one-time cost.

    Args:
        use_colbert: If True, include ColBERT multi-vector embeddings.
                     Set False for faster indexing when ColBERT reranking not needed.
    """

    def __init__(self, use_colbert: bool = True) -> None:
        self.use_colbert = use_colbert
        self._dense_model = None
        self._splade_model = None
        self._colbert_model = None

    def _get_dense(self):
        if self._dense_model is None:
            try:
                from fastembed import TextEmbedding
                self._dense_model = TextEmbedding(DENSE_MODEL)
                logger.info("Dense model loaded: %s", DENSE_MODEL)
            except Exception as exc:
                raise RuntimeError(f"Failed to load dense model {DENSE_MODEL}") from exc
        return self._dense_model

    def _get_splade(self):
        if self._splade_model is None:
            try:
                from fastembed import SparseTextEmbedding
                self._splade_model = SparseTextEmbedding(SPLADE_MODEL)
                logger.info("SPLADE model loaded: %s", SPLADE_MODEL)
            except Exception as exc:
                raise RuntimeError(f"Failed to load SPLADE model {SPLADE_MODEL}") from exc
        return self._splade_model

    def _get_colbert(self):
        if self._colbert_model is None:
            try:
                from fastembed import LateInteractionTextEmbedding
                self._colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)
                logger.info("ColBERT model loaded: %s", COLBERT_MODEL)
            except Exception as exc:
                raise RuntimeError(f"Failed to load ColBERT model {COLBERT_MODEL}") from exc
        return self._colbert_model

    def embed_document(self, text: str) -> EmbeddedVectors:
        """Embed a single document block — all three vector types."""
        return self.embed_documents_batch([text])[0]

    def embed_query(self, text: str) -> EmbeddedVectors:
        """Embed a query — uses query-specific ColBERT encoder."""
        dense_vec = list(next(self._get_dense().embed([text])).tolist())
        sparse_emb = next(self._get_splade().embed([text]))
        sparse_indices = sparse_emb.indices.tolist()
        sparse_values = sparse_emb.values.tolist()

        colbert_mat = None
        if self.use_colbert:
            raw = next(self._get_colbert().query_embed([text]))
            colbert_mat = raw.tolist()

        return EmbeddedVectors(
            dense=dense_vec,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            colbert=colbert_mat,
        )

    def embed_documents_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[EmbeddedVectors]:
        """Batch-embed documents efficiently — minimizes model overhead.

        Processes in batches of `batch_size` to avoid OOM on large corpora.
        Each model is run across all texts before moving to the next model,
        which is more cache-friendly than per-document embedding.
        """
        if not texts:
            return []

        logger.info("Embedding %d documents (batch_size=%d)", len(texts), batch_size)

        # Dense: all texts
        dense_vecs = list(self._get_dense().embed(texts, batch_size=batch_size))
        logger.debug("Dense embedding done: %d vectors", len(dense_vecs))

        # SPLADE sparse: all texts
        sparse_embs = list(self._get_splade().embed(texts, batch_size=batch_size))
        logger.debug("SPLADE embedding done: %d sparse vectors", len(sparse_embs))

        # ColBERT multivector: all texts (most memory-intensive — N×128 per doc)
        colbert_mats = None
        if self.use_colbert:
            colbert_mats = list(
                self._get_colbert().embed(texts, batch_size=min(batch_size, 16))
            )
            logger.debug("ColBERT embedding done: %d multivectors", len(colbert_mats))

        results: list[EmbeddedVectors] = []
        for i, (dense, sparse) in enumerate(zip(dense_vecs, sparse_embs, strict=True)):
            colbert = colbert_mats[i].tolist() if colbert_mats is not None else None
            results.append(EmbeddedVectors(
                dense=dense.tolist(),
                sparse_indices=sparse.indices.tolist(),
                sparse_values=sparse.values.tolist(),
                colbert=colbert,
            ))

        return results

    def is_available(self) -> bool:
        """True if all required models can be loaded."""
        try:
            self._get_dense()
            self._get_splade()
            if self.use_colbert:
                self._get_colbert()
            return True
        except Exception:
            return False
