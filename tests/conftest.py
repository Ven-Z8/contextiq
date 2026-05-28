"""Session-scoped fixtures for ContextIQ integration tests.

Key problems solved here:
1. Qdrant lock contention — local Qdrant only allows ONE client per path.
   All tests share a single LocalDocumentStore instance (session scope).
2. Legacy corpus has no hierarchy — blocks ingested before HierarchyBuilder
   was added have no parent_id / section_id metadata and no parents/ files.
   We rebuild hierarchy in-place at session start when parents/ is empty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from contextiq.retrieval.store import LocalDocumentStore

logger = logging.getLogger(__name__)

CORPUS_BLOCKS = Path("data/processed/blocks.json")
PARENTS_DIR = Path("data/processed/parents")


def _hierarchy_needs_rebuild(store: LocalDocumentStore) -> bool:
    """Return True if any indexed document is missing its parents file."""
    manifest = store._load_manifest()
    if not manifest:
        return False
    for doc_id in manifest:
        parent_path = store._parent_path(doc_id)
        if not parent_path.exists():
            return True
    return False


def _rebuild_hierarchy(store: LocalDocumentStore) -> None:
    """Re-run save_blocks for each document to tag blocks and write parents/."""
    manifest = store._load_manifest()
    logger.info("Rebuilding hierarchy for %d documents", len(manifest))
    for doc_id in manifest:
        blocks = store._load_document_blocks(doc_id)
        if not blocks:
            continue
        # save_blocks re-tags blocks with parent_id/section_id and writes parents/
        store.save_blocks(blocks)
        logger.info("Rebuilt hierarchy for %s: %d blocks", doc_id, len(blocks))


@pytest.fixture(scope="session")
def corpus_store() -> LocalDocumentStore:
    """Single shared store instance for all integration tests.

    Shared scope prevents Qdrant local-DB lock contention — only one
    QdrantClient is created for the entire test session.

    Also rebuilds hierarchy for legacy corpus that predates HierarchyBuilder.
    """
    if not CORPUS_BLOCKS.exists():
        pytest.skip("Processed corpus not found at data/processed/blocks.json")

    store = LocalDocumentStore(path=CORPUS_BLOCKS)

    if _hierarchy_needs_rebuild(store):
        logger.info("parents/ is stale or missing — rebuilding hierarchy")
        _rebuild_hierarchy(store)
        # Clear caches so rebuilt metadata is loaded fresh
        store._document_cache.clear()
        store._parent_cache.clear()

    return store


@pytest.fixture(scope="session")
def vector_available(corpus_store: LocalDocumentStore) -> bool:
    """True when the Qdrant vector index is accessible."""
    index = corpus_store._get_vector_index()
    return index is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_vector: skip test when local Qdrant index is unavailable",
    )
