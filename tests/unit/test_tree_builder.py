from __future__ import annotations

from contextiq.ingestion.tree import DocumentTree, TreeNode


def test_document_tree_round_trips_through_json() -> None:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0,
                    page_start=None, page_end=None, parent_id=None)
    tree = DocumentTree(document_id="d", source_path="d.pdf",
                        root_id="d:n0", nodes={"d:n0": root})

    restored = DocumentTree.model_validate_json(tree.model_dump_json())
    assert restored.root_id == "d:n0"
    assert restored.nodes["d:n0"].level == 0
