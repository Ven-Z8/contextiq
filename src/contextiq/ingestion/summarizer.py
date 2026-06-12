"""Section-local node summaries for the document tree."""

from __future__ import annotations

import logging

from contextiq.ingestion.tree import DocumentTree, TreeNode
from contextiq.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You summarize one section of a document so a retrieval agent can decide "
    "whether it is relevant. Output ONE or TWO sentences. No preamble."
)
_MAX_INPUT_CHARS = 6000


class NodeSummarizer:
    """Fill TreeNode.summary from each node's own block text."""

    def __init__(self, *, llm: LLMClient, min_words: int = 40,
                 max_tokens: int = 120) -> None:
        self.llm = llm
        self.min_words = min_words
        self.max_tokens = max_tokens

    def summarize(self, tree: DocumentTree, block_text: dict[str, str]) -> DocumentTree:
        for node in tree.nodes.values():
            text = self._node_text(node, block_text)
            if len(text.split()) < self.min_words:
                continue
            node.summary = self._summarize_one(node.title, text)
        return tree

    def _node_text(self, node: TreeNode, block_text: dict[str, str]) -> str:
        parts = [block_text.get(bid, "") for bid in node.block_ids]
        return "\n".join(p for p in parts if p).strip()

    def _summarize_one(self, title: str, text: str) -> str:
        prompt = f"Section title: {title or '(untitled)'}\n\n{text[:_MAX_INPUT_CHARS]}"
        try:
            return self.llm.generate(
                system_prompt=_SYSTEM, user_prompt=prompt, max_tokens=self.max_tokens
            ).text.strip()
        except Exception as exc:
            logger.warning("Node summary failed for %r: %s", title, exc)
            return ""
