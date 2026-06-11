"""Recursive document tree for reasoning-based navigation (sub-project #2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from contextiq.ingestion.models import BlockType, DocumentBlock


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


_MAX_DEPTH = 6


class TreeBuilder:
    """Build a recursive DocumentTree from heading-leveled blocks (top-down)."""

    def build(self, blocks: list[DocumentBlock]) -> DocumentTree:
        document_id = blocks[0].document_id if blocks else "document"
        source_path = blocks[0].source_path if blocks else ""
        nodes: dict[str, TreeNode] = {}
        counter = 0

        def new_node(title: str, level: int, parent_id: str | None) -> TreeNode:
            nonlocal counter
            node = TreeNode(
                node_id=f"{document_id}:n{counter}", document_id=document_id,
                title=title, level=level, parent_id=parent_id,
            )
            nodes[node.node_id] = node
            counter += 1
            return node

        root = new_node("", 0, None)
        stack: list[TreeNode] = [root]

        for block in blocks:
            if block.block_type == BlockType.HEADING:
                level = self._heading_level(block)
                level = max(1, min(level, _MAX_DEPTH))
                while len(stack) > level:
                    stack.pop()
                while len(stack) < level:
                    stack.append(stack[-1])
                parent = stack[-1]
                node = new_node(self._title(block), level, parent.node_id)
                parent.child_node_ids.append(node.node_id)
                stack = stack[:level] + [node]
            else:
                top = stack[-1]
                top.block_ids.append(block.block_id)
                self._extend_pages(top, block.page)

        self._roll_pages_up(root, nodes)
        return DocumentTree(
            document_id=document_id, source_path=source_path,
            root_id=root.node_id, nodes=nodes,
        )

    def _heading_level(self, block: DocumentBlock) -> int:
        meta = block.metadata.get("heading_level")
        if isinstance(meta, int):
            return meta
        stripped = block.text.lstrip()
        hashes = len(stripped) - len(stripped.lstrip("#"))
        return hashes or 1

    def _title(self, block: DocumentBlock) -> str:
        return block.text.lstrip("#").strip()

    def _extend_pages(self, node: TreeNode, page: int | None) -> None:
        if page is None:
            return
        node.page_start = page if node.page_start is None else min(node.page_start, page)
        node.page_end = page if node.page_end is None else max(node.page_end, page)

    def _roll_pages_up(self, node: TreeNode, nodes: dict[str, TreeNode]) -> None:
        for child_id in node.child_node_ids:
            child = nodes[child_id]
            self._roll_pages_up(child, nodes)
            self._extend_pages(node, child.page_start)
            self._extend_pages(node, child.page_end)
