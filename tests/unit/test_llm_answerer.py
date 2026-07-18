from __future__ import annotations

from types import SimpleNamespace

from contextiq.context.models import ContextPacket, ContextSource
from contextiq.core.config import Settings
from contextiq.ingestion.models import DocumentBlock
from contextiq.llm.answerer import GroundedAnswerer
from contextiq.llm.client import AnthropicLLMClient, LLMClient, LLMResult
from contextiq.llm.prompts import load_answer_prompt


class FakeLLMClient(LLMClient):
    def __init__(self, answer_text: str = "Grounded answer [doc:1, page 3].") -> None:
        self.user_prompt = ""
        self.answer_text = answer_text

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> LLMResult:
        del system_prompt, max_tokens
        self.user_prompt = user_prompt
        return LLMResult(
            text=self.answer_text,
            model="fake-model",
            mode="test",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0,
        )


def test_answer_prompt_loads_from_prompt_file() -> None:
    prompt = load_answer_prompt()

    assert "enterprise document intelligence" in prompt.system
    assert "visual evidence" in prompt.system
    assert "Docling visual description" in prompt.system
    assert "partially readable" in prompt.system
    assert "{{question}}" in prompt.user
    assert "{{context_packet}}" in prompt.user


def test_grounded_answerer_sends_context_packet_to_client() -> None:
    client = FakeLLMClient()
    packet = ContextPacket(
        question="What regulatory risks exist?",
        sources=[
            ContextSource(
                block=DocumentBlock(
                    document_id="doc",
                    block_id="doc:1",
                    source_path="sample.md",
                    page=3,
                    text="Regulatory investigations create risk.",
                ),
                estimated_tokens=12,
                reason="lexical selected this block",
                stages=["lexical"],
                score=9.0,
            )
        ],
        token_budget=1000,
        used_tokens=12,
        dropped_candidates=0,
    )

    answer = GroundedAnswerer(client=client, settings=Settings()).answer(packet)

    assert answer.text == "Grounded answer [doc:1, page 3]."
    assert "What regulatory risks exist?" in client.user_prompt
    assert "doc:1" in client.user_prompt


def test_grounded_answerer_warns_when_answer_infers_missing_page() -> None:
    client = FakeLLMClient("The answer is supported on page 6.")
    packet = ContextPacket(
        question="What does the table say?",
        sources=[
            ContextSource(
                block=DocumentBlock(
                    document_id="doc",
                    block_id="doc:6:chunk-0",
                    source_path="sample.xlsx",
                    page=None,
                    text="A table row.",
                ),
                estimated_tokens=4,
                reason="asset_mapping selected this block",
            )
        ],
        token_budget=1000,
        used_tokens=4,
        dropped_candidates=0,
    )

    answer = GroundedAnswerer(client=client, settings=Settings()).answer(packet)

    assert answer.warnings == [
        "Answer mentions a numeric page, but retrieved sources only report page unknown."
    ]


def test_grounded_answerer_warns_for_fabricated_citation() -> None:
    client = FakeLLMClient(
        "Unsupported claim [doc:missing, page 7] and wrong page [doc:1, page 8]."
    )
    packet = ContextPacket(
        question="What does the filing say?",
        sources=[
            ContextSource(
                block=DocumentBlock(
                    document_id="doc",
                    block_id="doc:1",
                    source_path="sample.pdf",
                    page=3,
                    text="Supported evidence.",
                ),
                estimated_tokens=4,
                reason="retrieved this block",
            )
        ],
        token_budget=1000,
        used_tokens=4,
        dropped_candidates=0,
    )

    answer = GroundedAnswerer(client=client, settings=Settings()).answer(packet)

    assert answer.warnings == [
        "Citation references unknown block 'doc:missing'.",
        "Citation for block 'doc:1' reports page 8, but the retrieved source is page 3.",
    ]


def test_context_packet_renders_page_zero() -> None:
    packet = ContextPacket(
        question="What does the source say?",
        sources=[
            ContextSource(
                block=DocumentBlock(
                    document_id="doc",
                    block_id="doc:0",
                    source_path="sample.pdf",
                    page=0,
                    text="First page.",
                ),
                estimated_tokens=3,
                reason="retrieved this block",
            )
        ],
        token_budget=1000,
        used_tokens=3,
        dropped_candidates=0,
    )

    assert "### doc:0 (page 0)" in packet.to_markdown()


def test_anthropic_client_warns_when_max_tokens_truncates(monkeypatch) -> None:
    client = AnthropicLLMClient(api_key="test", model="claude-test")
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Incomplete answer")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="max_tokens",
    )
    monkeypatch.setattr(client.client.messages, "create", lambda **kwargs: message)

    result = client.generate(system_prompt="s", user_prompt="u", max_tokens=5)

    assert result.warnings == [
        "Answer may be truncated because the model reached the maximum token limit."
    ]
