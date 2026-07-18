"""Context packet models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from contextiq.ingestion.models import BlockType, DocumentBlock


class ContextSource(BaseModel):
    """A selected source in a context packet."""

    block: DocumentBlock
    estimated_tokens: int
    reason: str
    stages: list[str] = Field(default_factory=list)
    score: float | None = None


class ContextPacket(BaseModel):
    """Packed context and its quality report."""

    question: str
    sources: list[ContextSource] = Field(default_factory=list)
    token_budget: int
    used_tokens: int
    dropped_candidates: int

    def to_markdown(self) -> str:
        lines = [
            "# Context Packet",
            "",
            f"Question: {self.question}",
            f"Token budget: {self.token_budget}",
            f"Used tokens: {self.used_tokens}",
            f"Dropped candidates: {self.dropped_candidates}",
            "",
            "## Sources",
        ]
        for source in self.sources:
            block = source.block
            page = f"page {block.page}" if block.page is not None else "page unknown"
            lines.extend(
                [
                    "",
                    f"### {block.block_id} ({page})",
                    f"Reason: {source.reason}",
                    f"Trace: stages={', '.join(source.stages) or 'unknown'} score={source.score}",
                    *self._format_visual_metadata(block),
                    self._format_block_text(block),
                ]
            )
        return "\n".join(lines)

    def _format_visual_metadata(self, block: DocumentBlock) -> list[str]:
        metadata = block.metadata
        visual_kind = metadata.get("visual_kind")
        if not visual_kind:
            return []

        lines = [f"Visual: {visual_kind}"]
        if caption := metadata.get("caption"):
            lines.append(f"Caption: {caption}")
        if visual_class := metadata.get("visual_class"):
            confidence = metadata.get("visual_class_confidence")
            if confidence is not None:
                lines.append(f"Visual class: {visual_class} (confidence {confidence})")
            else:
                lines.append(f"Visual class: {visual_class}")
        if visual_description := metadata.get("visual_description"):
            lines.append(f"Visual description: {visual_description}")
        if provider := metadata.get("visual_description_provider"):
            lines.append(f"Visual description provider: {provider}")
        if image_path := metadata.get("image_path"):
            lines.append(f"Image artifact: {image_path}")
        return lines

    def _format_block_text(self, block: DocumentBlock) -> str:
        if block.block_type == BlockType.TABLE:
            return f"```text\n{block.text}\n```"
        return block.text
