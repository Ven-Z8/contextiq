from __future__ import annotations

from typing import Any

from contextiq.ui import gradio_app


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_ask_question_returns_answer_and_filtered_source_trace(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **_kwargs: Any):
        captured["url"] = url
        return FakeResponse(
            {
                "question": "Find the revenue chart",
                "answer": "Services revenue grew. [doc:figure, page 7]",
                "model": "test-model",
                "mode": "openrouter",
                "tokens_in": 120,
                "tokens_out": 40,
                "cost_usd": None,
                "warnings": [],
                "context": {
                    "question": "Find the revenue chart",
                    "markdown": "# Context Packet",
                    "token_budget": 1000,
                    "used_tokens": 220,
                    "dropped_candidates": 0,
                    "sources": [
                        {
                            "block_id": "doc:figure",
                            "block_type": "figure",
                            "page": 7,
                            "section_path": ["Revenue"],
                            "stages": ["lexical", "expansion"],
                            "score": 14.25,
                            "reason": "lexical selected this figure block",
                            "metadata": {
                                "caption": "Revenue by segment chart",
                                "visual_kind": "figure",
                                "visual_class": "bar_chart",
                                "visual_description": "Services revenue grew faster.",
                                "image_path": "data/processed/visuals/doc/block-00001.png",
                            },
                            "text": "Figure: Revenue by segment chart.",
                        },
                        {
                            "block_id": "doc:text",
                            "block_type": "text",
                            "page": 8,
                            "section_path": ["Revenue"],
                            "stages": ["vector"],
                            "score": 3.0,
                            "reason": "vector selected this block",
                            "metadata": {},
                            "text": "Text source.",
                        },
                    ],
                },
            }
        )

    monkeypatch.setattr(gradio_app.httpx, "post", fake_post)

    answer, context, sources = gradio_app.ask_question(
        question="Find the revenue chart",
        backend_url="http://backend",
        token_budget=1000,
        limit=5,
        block_types=["figure"],
    )

    assert captured["url"] == "http://backend/answer"
    assert context == "# Context Packet"
    assert "Services revenue grew." in answer
    assert "page 7" in answer
    assert "doc:figure" in answer
    assert "cited" in answer
    assert "doc:figure" in sources
    assert "cited in answer" in sources
    assert "doc:text" not in sources
    assert (
        "http://backend/visuals?path=data/processed/visuals/doc/block-00001.png"
        in sources
    )


def test_ask_question_empty_question_returns_three_panels() -> None:
    answer, context, sources = gradio_app.ask_question("", "", 1000, 5, ["text"])

    assert "Ask a question first." in answer
    assert context == "Ask a question first."
    assert sources == ""


def test_format_source_cards_renders_compact_html_table_evidence() -> None:
    sources = gradio_app._format_source_cards(
        [
            {
                "block_id": "nasa:uc",
                "block_type": "table",
                "page": None,
                "section_path": ["Use Cases"],
                "stages": ["structured_code"],
                "score": 73.0,
                "reason": "structured code selected this table evidence block",
                "metadata": {"visual_kind": "spreadsheet"},
                "text": (
                    "| UC ID | Use Cases |\n"
                    "| --- | --- |\n"
                    "| UC-T-202 L | Transportation of large cargo |"
                ),
            }
        ],
        ["table"],
    )

    assert "source-card" in sources
    assert "structured_code" in sources
    assert "UC-T-202 L" in sources
    assert "source-table" in sources


def test_cited_block_ids_parses_page_citations() -> None:
    ids = gradio_app._cited_block_ids(
        "See [AMD_2022_10K-abc:583, page 60] and also [doc:text]."
    )
    assert ids == {"AMD_2022_10K-abc:583", "doc:text"}
