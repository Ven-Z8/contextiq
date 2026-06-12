"""Filesystem persistence for DocumentTree JSON."""

from __future__ import annotations

import re
from pathlib import Path

from contextiq.ingestion.tree import DocumentTree

_SLUG = re.compile(r"[^a-zA-Z0-9_.-]+")


class TreeStore:
    """Read/write DocumentTree JSON under a trees directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("data/processed/trees")

    def _path(self, document_id: str) -> Path:
        slug = _SLUG.sub("-", document_id).strip("-") or "document"
        return self.root / f"{slug[:120]}.json"

    def save(self, tree: DocumentTree) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(tree.document_id)
        path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, document_id: str) -> DocumentTree | None:
        path = self._path(document_id)
        if not path.exists():
            return None
        return DocumentTree.model_validate_json(path.read_text(encoding="utf-8"))
