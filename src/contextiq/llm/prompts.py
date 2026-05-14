"""Prompt loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnswerPrompt:
    """Prompt template for answer synthesis."""

    system: str
    user: str

    def render_user(self, *, question: str, context_packet: str) -> str:
        """Render the user prompt with runtime values."""

        return self.user.replace("{{question}}", question).replace(
            "{{context_packet}}",
            context_packet,
        )


def load_answer_prompt(path: Path | None = None) -> AnswerPrompt:
    """Load the answer prompt from the project prompts directory."""

    prompt_path = path or Path("prompts/answer.yaml")
    text = prompt_path.read_text(encoding="utf-8")
    return AnswerPrompt(
        system=_extract_literal_block(text=text, key="system"),
        user=_extract_literal_block(text=text, key="user"),
    )


def _extract_literal_block(*, text: str, key: str) -> str:
    marker = f"{key}: |"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != marker:
            continue
        block_lines: list[str] = []
        for block_line in lines[index + 1 :]:
            if block_line and not block_line.startswith(" "):
                break
            block_lines.append(block_line[2:] if block_line.startswith("  ") else block_line)
        return "\n".join(block_lines).strip()
    raise ValueError(f"Prompt file is missing {key!r} literal block")
