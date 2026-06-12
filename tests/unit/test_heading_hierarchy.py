from __future__ import annotations

from contextiq.ingestion.heading_hierarchy import HeadingHierarchyInferencer
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.tree import TreeBuilder
from contextiq.llm.client import LLMClient, LLMResult


def _h(idx: int, title: str) -> DocumentBlock:
    return DocumentBlock(
        document_id="d", block_id=f"d:{idx}", source_path="x.pdf",
        block_type=BlockType.HEADING, text=title,
        metadata={"reading_order": idx, "heading_level": 1},
    )


def _b(idx: int, text: str) -> DocumentBlock:
    return DocumentBlock(
        document_id="d", block_id=f"d:{idx}", source_path="x.pdf",
        block_type=BlockType.TEXT, text=text, metadata={"reading_order": idx},
    )


def test_heuristic_assigns_part_item_and_numbered_levels() -> None:
    inf = HeadingHierarchyInferencer(llm=None)
    blocks = [_h(0, "PART I"), _h(1, "Item 1. Business"), _h(2, "Company Background"),
              _h(3, "1.2.1 Detail"), _b(4, "body text")]
    out = inf.assign(blocks)
    lvl = {b.text: b.metadata["heading_level"] for b in out if b.block_type == BlockType.HEADING}
    assert lvl["PART I"] == 1
    assert lvl["Item 1. Business"] == 2
    assert lvl["Company Background"] == 3
    assert lvl["1.2.1 Detail"] == 3
    assert out[4].metadata.get("heading_level") is None  # non-heading untouched


def test_llm_levels_applied_when_client_returns_json() -> None:
    class FakeLLM(LLMClient):
        def generate(self, *, system_prompt, user_prompt, max_tokens) -> LLMResult:
            return LLMResult(text="[1, 2, 2]", model="f", mode="f", tokens_in=1, tokens_out=1)

    out = HeadingHierarchyInferencer(llm=FakeLLM()).assign([_h(0, "A"), _h(1, "B"), _h(2, "C")])
    assert [b.metadata["heading_level"] for b in out] == [1, 2, 2]


def test_llm_failure_falls_back_to_heuristic_and_never_raises() -> None:
    class Boom(LLMClient):
        def generate(self, **_):
            raise RuntimeError("api down")

    inf = HeadingHierarchyInferencer(llm=Boom())
    out = inf.assign([_h(0, "PART I"), _h(1, "Item 1. Business")])
    assert [b.metadata["heading_level"] for b in out] == [1, 2]


def test_llm_wrong_length_output_falls_back_to_heuristic() -> None:
    class Bad(LLMClient):
        def generate(self, **_) -> LLMResult:
            return LLMResult(text="[1]", model="f", mode="f", tokens_in=1, tokens_out=1)

    out = HeadingHierarchyInferencer(llm=Bad()).assign([_h(0, "PART I"), _h(1, "Item 1. Business")])
    assert [b.metadata["heading_level"] for b in out] == [1, 2]


def test_inference_makes_the_tree_hierarchical_not_flat() -> None:
    inf = HeadingHierarchyInferencer(llm=None)
    blocks = [_h(0, "PART I"), _h(1, "Item 1. Business"), _b(2, "x"),
              _h(3, "Company Background"), _b(4, "y")]
    tree = TreeBuilder().build(inf.assign(blocks))
    levels = {n.level for n in tree.nodes.values()}
    assert max(levels) >= 3  # PART(1) > Item(2) > Company(3), no longer flat
