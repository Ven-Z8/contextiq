"""promptfoo provider for contextiq."""

from __future__ import annotations

from contextiq.query import answer_question
from ven_eval import EvalOutput, PromptfooProvider


class ContextiqProvider(PromptfooProvider):
    def run(self, question: str) -> EvalOutput:
        answer, packet = answer_question(question)
        metadata = _extract_metadata(answer, packet)
        return EvalOutput(
            answer=answer.text,
            metadata=metadata,
            cost_usd=answer.cost_usd,
            model=answer.model,
            token_usage={"prompt": answer.tokens_in, "completion": answer.tokens_out},
        )


def _extract_metadata(answer, packet) -> dict:
    source_ids = [s.block.block_id for s in packet.sources]
    return {
        "model": answer.model,
        "warnings": answer.warnings,
        "citations_count": len(packet.sources),
        "source_ids": source_ids,
        "budget": {
            "token_budget": packet.token_budget,
            "used_tokens": packet.used_tokens,
            "dropped_candidates": packet.dropped_candidates,
            "selected_source_ids": source_ids,
        },
    }


_provider = ContextiqProvider()


def call_api(prompt, options, context):
    return _provider.call_api(prompt, options, context)
