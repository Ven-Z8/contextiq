"""Agentic retrieve: decompose + merge/dedup + rerank (fakes, no API)."""

from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.llm.client import LLMResult
from contextiq.retrieval.agentic import (
    _extract_json_array,
    _parse_index,
    agentic_retrieve,
    decompose,
    route_document,
)
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
        self._scoped_document_id = "scoped"  # already scoped -> routing skipped

    def hybrid_hits(self, query: str, limit: int) -> list[RetrievalHit]:
        return self._hits  # same pool for each sub-query -> exercises dedup


class _FakeManifestStore:
    def __init__(self, docs: list[str]) -> None:
        self._docs = docs

    def _load_manifest(self) -> list[str]:
        return self._docs


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
    assert client.calls == 2  # scoped store -> one decompose + one rerank, no router


def test_parse_index_validates_range() -> None:
    assert _parse_index("1", 3) == 1
    assert _parse_index("the filing is 2", 3) == 2
    assert _parse_index("-1", 3) == -1
    assert _parse_index("99", 3) == -1  # out of range -> -1


def test_route_document_picks_target() -> None:
    store = _FakeManifestStore(["AMD_2022_10K-x", "BOEING_2022_10K-y"])
    client = _FakeClient("1", "")  # router -> index 1
    assert route_document(store, "What was Boeing revenue?", client) == "BOEING_2022_10K-y"


def test_route_document_single_doc_skips_llm() -> None:
    store = _FakeManifestStore(["ONLY_10K-x"])
    client = _FakeClient("", "")
    assert route_document(store, "q", client) == "ONLY_10K-x"
    assert client.calls == 0  # no LLM call when there is only one filing
