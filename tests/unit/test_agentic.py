"""Agentic retrieve: decompose + merge/dedup + rerank (fakes, no API)."""

from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.llm.client import LLMResult
from contextiq.retrieval.agentic import _extract_json_array, agentic_retrieve, decompose
from contextiq.retrieval.models import RetrievalHit


def _hit(block_id: str, text: str, score: float) -> RetrievalHit:
    block = DocumentBlock(
        document_id="d",
        block_id=block_id,
        source_path="x",
        page=1,
        section_path=[],
        block_type=BlockType.TEXT,
        text=text,
        metadata={},
    )
    return RetrievalHit(block=block, rank=0, score=score, stages=["hybrid"], reason="")


class _FakeClient:
    def __init__(self, decompose_json: str, rerank_json: str) -> None:
        self._decompose_json = decompose_json
        self._rerank_json = rerank_json
        self.calls = 0

    def generate(self, *, system_prompt: str, user_prompt: str, max_tokens: int) -> LLMResult:
        self.calls += 1
        text = self._decompose_json if self.calls == 1 else self._rerank_json
        return LLMResult(text=text, model="fake", mode="fake", tokens_in=1, tokens_out=1)


class _FakeStore:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    def hybrid_hits(self, query: str, limit: int) -> list[RetrievalHit]:
        return self._hits  # same pool for each sub-query -> exercises dedup


def test_extract_json_array_ignores_surrounding_prose() -> None:
    assert _extract_json_array('reasoning... ["a","b"] done') == '["a","b"]'
    assert _extract_json_array("no array here") == "[]"


def test_decompose_parses_line_items() -> None:
    client = _FakeClient('["current assets","current liabilities"]', "[]")
    assert decompose(client, "quick ratio?") == ["current assets", "current liabilities"]


def test_agentic_merges_dedups_and_reranks() -> None:
    hits = [_hit("d:0", "alpha", 5.0), _hit("d:1", "beta", 3.0), _hit("d:2", "gamma", 4.0)]
    client = _FakeClient('["q1","q2"]', "[2, 0]")  # rerank -> d:2 then d:0
    out = agentic_retrieve(_FakeStore(hits), "q", client, k=2, per_query=12)
    assert [h.block.block_id for h in out] == ["d:2", "d:0"]
    assert client.calls == 2  # exactly one decompose + one rerank
