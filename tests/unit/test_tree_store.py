from __future__ import annotations

from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.ingestion.tree_store import TreeStore


def _tree() -> DocumentTree:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0, parent_id=None)
    return DocumentTree(document_id="d", source_path="d.pdf", root_id="d:n0",
                        nodes={"d:n0": root})


def test_tree_store_round_trips(tmp_path) -> None:
    store = TreeStore(root=tmp_path / "trees")
    store.save(_tree())
    loaded = store.load("d")
    assert loaded is not None
    assert loaded.root_id == "d:n0"


def test_tree_store_returns_none_for_missing(tmp_path) -> None:
    assert TreeStore(root=tmp_path / "trees").load("nope") is None
