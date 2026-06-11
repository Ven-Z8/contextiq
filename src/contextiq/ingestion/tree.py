"""Recursive document tree for reasoning-based navigation (sub-project #2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    """One node in the document tree (document root, section, or subsection)."""

    node_id: str
    document_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    summary: str = ""
    parent_id: str | None = None
    child_node_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)


class DocumentTree(BaseModel):
    """A document parsed into a recursive node hierarchy."""

    document_id: str
    source_path: str
    root_id: str
    nodes: dict[str, TreeNode]
    page_count: int | None = None
