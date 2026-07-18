from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from contextiq.api.main import create_app
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.llm.answerer import GroundedAnswerer
from contextiq.llm.client import ExtractiveFallbackClient
from contextiq.retrieval.store import LocalDocumentStore


def test_api_health() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_serves_lightweight_dashboard() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ContextIQ" in response.text
    assert "buildContext" in response.text
    assert "answerQuestion" in response.text
    assert "/answer" in response.text
    assert "/context" in response.text


def test_api_context_packet(tmp_path: Path) -> None:
    store_path = tmp_path / "blocks.json"
    store = LocalDocumentStore(path=store_path)
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                page=3,
                text=(
                    "Competition regulations and government investigations create "
                    "regulatory risk."
                ),
            )
        ]
    )
    client = TestClient(
        create_app(
            store_path=store_path,
            answerer_factory=lambda: GroundedAnswerer(client=ExtractiveFallbackClient()),
        )
    )

    response = client.post(
        "/context",
        json={"question": "What regulatory risks exist?", "token_budget": 1000, "limit": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"][0]["page"] == 3
    assert "doc:1" in payload["markdown"]


def test_api_context_sources_include_retrieval_trace_and_metadata(tmp_path: Path) -> None:
    store_path = tmp_path / "blocks.json"
    store = LocalDocumentStore(path=store_path)
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:figure",
                source_path="sample.md",
                block_type=BlockType.FIGURE,
                text="Figure: Revenue by segment chart.",
                metadata={"visual_kind": "figure", "caption": "Revenue by segment chart"},
            )
        ]
    )
    client = TestClient(
        create_app(
            store_path=store_path,
            answerer_factory=lambda: GroundedAnswerer(client=ExtractiveFallbackClient()),
        )
    )

    response = client.post(
        "/context",
        json={
            "question": "Find the revenue by segment chart",
            "token_budget": 1000,
            "limit": 3,
        },
    )

    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["block_type"] == "figure"
    assert "lexical" in source["stages"]
    assert source["score"] > 0
    assert source["metadata"]["caption"] == "Revenue by segment chart"


def test_api_serves_visual_artifacts_from_processed_visuals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = Path("data/processed/visuals/doc/block-00001.png")
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="white").save(image_path)
    client = TestClient(create_app())

    response = client.get("/visuals", params={"path": str(image_path)})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content


def test_api_rejects_visual_artifact_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())

    response = client.get("/visuals", params={"path": "../secret.png"})

    assert response.status_code == 404


def test_api_answer_uses_fallback_without_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXTIQ_ANTHROPIC_API_KEY", raising=False)
    store_path = tmp_path / "blocks.json"
    store = LocalDocumentStore(path=store_path)
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                page=3,
                text="Regulatory investigations create risk.",
            )
        ]
    )
    client = TestClient(
        create_app(
            store_path=store_path,
            answerer_factory=lambda: GroundedAnswerer(client=ExtractiveFallbackClient()),
        )
    )

    response = client.post(
        "/answer",
        json={"question": "What regulatory risks exist?", "token_budget": 1000, "limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "extractive_fallback"
    assert "no LLM API key" in payload["answer"]
    assert payload["context"]["sources"][0]["block_id"] == "doc:1"
