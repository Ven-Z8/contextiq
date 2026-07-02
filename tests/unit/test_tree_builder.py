from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.tree import DocumentTree, TreeBuilder, TreeNode


def test_document_tree_round_trips_through_json() -> None:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0,
                    page_start=None, page_end=None, parent_id=None)
    tree = DocumentTree(document_id="d", source_path="d.pdf",
                        root_id="d:n0", nodes={"d:n0": root})

    restored = DocumentTree.model_validate_json(tree.model_dump_json())
    assert restored.root_id == "d:n0"
    assert restored.nodes["d:n0"].level == 0


def _heading(doc, idx, level, text, page):
    return DocumentBlock(
        document_id=doc, block_id=f"{doc}:{idx}", source_path="d.pdf", page=page,
        block_type=BlockType.HEADING, text=text,
        metadata={"reading_order": idx, "heading_level": level},
    )


def _body(doc, idx, text, page):
    return DocumentBlock(
        document_id=doc, block_id=f"{doc}:{idx}", source_path="d.pdf", page=page,
        block_type=BlockType.TEXT, text=text, metadata={"reading_order": idx},
    )


def test_tree_builder_nests_h2_under_h1_and_attaches_blocks() -> None:
    blocks = [
        _heading("d", 0, 1, "Part I", 1),
        _heading("d", 1, 2, "Risk Factors", 2),
        _body("d", 2, "Risks here.", 2),
        _heading("d", 3, 2, "Properties", 5),
        _body("d", 4, "Buildings.", 5),
    ]
    tree = TreeBuilder().build(blocks)

    root = tree.nodes[tree.root_id]
    assert root.level == 0
    part = tree.nodes[root.child_node_ids[0]]
    assert part.title == "Part I"
    assert [tree.nodes[c].title for c in part.child_node_ids] == ["Risk Factors", "Properties"]
    risk = tree.nodes[part.child_node_ids[0]]
    assert risk.block_ids == ["d:2"]
    assert risk.page_start == 2


def test_tree_builder_rolls_page_ranges_up_to_parents() -> None:
    blocks = [
        _heading("d", 0, 1, "Part I", 1),
        _body("d", 1, "a", 1),
        _heading("d", 2, 2, "Sub", 3),
        _body("d", 3, "b", 9),
    ]
    tree = TreeBuilder().build(blocks)
    part = tree.nodes[tree.nodes[tree.root_id].child_node_ids[0]]
    assert part.page_start == 1
    assert part.page_end == 9


def test_tree_builder_headingless_doc_yields_single_root_with_all_blocks() -> None:
    blocks = [_body("d", 0, "x", 1), _body("d", 1, "y", 2)]
    tree = TreeBuilder().build(blocks)
    assert list(tree.nodes) == [tree.root_id]
    assert tree.nodes[tree.root_id].block_ids == ["d:0", "d:1"]
