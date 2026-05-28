#!/usr/bin/env python3
"""Re-index the entire ContextIQ corpus into the enterprise 3-vector collection.

Usage:
    uv run python scripts/reindex_enterprise.py
    uv run python scripts/reindex_enterprise.py --doc-id apple-2025-10k-e8c0f8fec7
    uv run python scripts/reindex_enterprise.py --no-colbert   # Skip ColBERT (faster)
    uv run python scripts/reindex_enterprise.py --dry-run      # Show plan, no indexing

What this does:
    1. Loads existing corpus from data/processed/blocks.json
    2. Applies AdaptiveChunker to re-chunk each block by content profile
    3. Re-runs HierarchyBuilder to tag parent_id / section_id on new chunks
    4. Indexes into the enterprise collection (SPLADE + ColBERT + dense)

The enterprise collection is separate from the legacy "contextiq_chunks_v2" collection.
Legacy search still works; enterprise search requires this script to have been run.

Expected time:
    Apple 10-K (939 blocks):   ~8-12 min  (ColBERT model download ~440 MB first time)
    NASA Lunar (219 blocks):   ~2-3 min
    NVIDIA Annual (2736 blocks): ~20-30 min
    MSFT 10-K (3303 blocks):   ~25-35 min
    Full corpus (7197 blocks): ~60-90 min

The SPLADE and ColBERT models are cached to ~/.cache/fastembed/ after first download.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextiq.ingestion.adaptive_chunker import AdaptiveChunker
from contextiq.retrieval.store import LocalDocumentStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reindex_enterprise")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-index corpus into enterprise 3-vector collection"
    )
    parser.add_argument(
        "--doc-id",
        help="Re-index only this document ID (default: all documents)",
    )
    parser.add_argument(
        "--no-colbert",
        action="store_true",
        help="Skip ColBERT embeddings (faster; reranking disabled)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Skip adaptive chunking (use existing blocks as-is)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (default: 16; reduce if OOM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/processed/blocks.json",
        help="Path to blocks.json manifest (default: data/processed/blocks.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        sys.exit(1)

    logger.info("Loading corpus from %s", corpus_path)
    store = LocalDocumentStore(path=corpus_path)
    manifest = store._load_manifest()

    if not manifest:
        logger.error("No documents in manifest. Run ingestion first.")
        sys.exit(1)

    logger.info("Corpus manifest: %d documents", len(manifest))
    for doc_id in manifest:
        logger.info("  • %s", doc_id)

    # Filter to specific doc if requested
    target_ids = manifest
    if args.doc_id:
        target_ids = [d for d in manifest if args.doc_id in d]
        if not target_ids:
            logger.error("No document matching --doc-id '%s'", args.doc_id)
            sys.exit(1)
        logger.info("Targeting: %s", target_ids)

    # Adaptive chunker setup
    chunker = AdaptiveChunker() if not args.no_adaptive else None

    # Enterprise EmbeddingPipeline
    if not args.dry_run:
        from contextiq.ingestion.embedding_pipeline import EmbeddingPipeline
        logger.info("Loading embedding models (first run downloads ~1 GB)...")
        pipeline = EmbeddingPipeline(use_colbert=not args.no_colbert)
        if not pipeline.is_available():
            logger.error("Embedding pipeline models failed to load")
            sys.exit(1)
        logger.info(
            "Models ready (dense=%s, SPLADE=%s, ColBERT=%s)",
            "BAAI/bge-small-en",
            "prithivida/Splade_PP_en_v1",
            "OFF" if args.no_colbert else "colbert-ir/colbertv2.0",
        )

    total_indexed = 0
    session_start = time.time()

    for doc_id in target_ids:
        logger.info("─" * 60)
        logger.info("Processing: %s", doc_id)
        t0 = time.time()

        # Load raw blocks
        blocks = store._load_document_blocks(doc_id)
        if not blocks:
            logger.warning("No blocks found for %s — skipping", doc_id)
            continue
        logger.info("  Raw blocks: %d", len(blocks))

        # Adaptive chunking
        if chunker is not None:
            blocks = chunker.process_blocks(blocks)
            profile_counts: dict[str, int] = {}
            for b in blocks:
                p = b.metadata.get("content_profile", "unknown")
                profile_counts[p] = profile_counts.get(p, 0) + 1
            logger.info("  After adaptive chunking: %d blocks", len(blocks))
            for profile, count in sorted(profile_counts.items()):
                logger.info("    %-20s: %d blocks", profile, count)

        if args.dry_run:
            logger.info("  [DRY RUN] Would index %d blocks", len(blocks))
            continue

        # Index into enterprise collection
        indexed = store.index_blocks_enterprise(blocks)
        elapsed = time.time() - t0
        logger.info(
            "  Indexed: %d blocks in %.1fs (%.1f blocks/sec)",
            indexed, elapsed, indexed / max(elapsed, 0.001),
        )
        total_indexed += indexed

    # Summary
    total_elapsed = time.time() - session_start
    logger.info("─" * 60)
    if args.dry_run:
        logger.info("DRY RUN complete. Use without --dry-run to index.")
    else:
        logger.info(
            "Enterprise indexing complete: %d blocks across %d documents in %.1fs",
            total_indexed, len(target_ids), total_elapsed,
        )

        # Verify enterprise collection
        status = store.enterprise_status()
        logger.info("Enterprise collection status:")
        for key, val in status.items():
            logger.info("  %s: %s", key, val)

        logger.info(
            "\nEnterprise search is now available via:\n"
            "  store.search_enterprise(query, limit=10)\n"
            "  vector_index.search_hybrid_with_reranking(query, limit=10)"
        )


if __name__ == "__main__":
    main()
