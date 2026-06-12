from __future__ import annotations

from contextiq.ingestion.summarizer import NodeSummarizer
from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.llm.client import LLMClient, LLMResult


class _FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system_prompt, user_prompt, max_tokens) -> LLMResult:
        self.calls += 1
        return LLMResult(text="A short section summary.", model="fake",
                         mode="fake", tokens_in=1, tokens_out=1)


def _tree_with_one_section() -> tuple[DocumentTree, dict[str, str]]:
    root = TreeNode(node_id="d:n0", document_id="d", title="", level=0, parent_id=None,
                    child_node_ids=["d:n1"])
    sec = TreeNode(node_id="d:n1", document_id="d", title="Risk Factors", level=1,
                   parent_id="d:n0", block_ids=["d:2"])
    tree = DocumentTree(document_id="d", source_path="d.pdf", root_id="d:n0",
                        nodes={"d:n0": root, "d:n1": sec})
    block_text = {"d:2": "A long paragraph about regulatory and market risks " * 10}
    return tree, block_text


def test_summarizer_sets_summary_on_content_nodes() -> None:
    tree, block_text = _tree_with_one_section()
    llm = _FakeLLM()
    NodeSummarizer(llm=llm, min_words=20).summarize(tree, block_text)
    assert tree.nodes["d:n1"].summary == "A short section summary."
    assert llm.calls == 1  # root (empty) skipped


def test_summarizer_skips_small_nodes() -> None:
    tree, _ = _tree_with_one_section()
    llm = _FakeLLM()
    NodeSummarizer(llm=llm, min_words=20).summarize(tree, {"d:2": "tiny"})
    assert tree.nodes["d:n1"].summary == ""
    assert llm.calls == 0


def test_summarizer_never_raises_on_llm_failure() -> None:
    tree, block_text = _tree_with_one_section()

    class _Boom(LLMClient):
        def generate(self, **_):
            raise RuntimeError("api down")

    NodeSummarizer(llm=_Boom(), min_words=20).summarize(tree, block_text)
    assert tree.nodes["d:n1"].summary == ""
