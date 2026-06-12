"""Infer heading nesting levels.

Extractors (Docling standard *and* granite-docling VLM) emit every heading as
`section_header_level_1` — they do not provide a real heading hierarchy (verified
on real 10-K output, issue #10). Without inferred levels the document tree is flat,
which defeats reasoning-based navigation. This module assigns nesting levels from
an ordered heading list, using an LLM when available and deterministic heuristics
as a fallback (and when no API key is configured).
"""

from __future__ import annotations

import json
import logging
import re

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.llm.client import LLMClient

logger = logging.getLogger(__name__)

_PART = re.compile(r"^\s*part\s+(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)
_ITEM = re.compile(r"^\s*item\s+\d+", re.IGNORECASE)
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\s|\.|$)")
_MAX_LEVEL = 6
_DEFAULT_MAX_LLM_HEADINGS = 250

_SYSTEM = (
    "You assign outline nesting levels to a document's headings. Given an ordered, "
    "numbered list of headings, reply with ONLY a JSON array of integers — one per "
    "heading, in the same order — where 1 is the top level and larger numbers nest "
    "deeper. No prose, no code fences."
)


class HeadingHierarchyInferencer:
    """Assign `metadata['heading_level']` to heading blocks via LLM + heuristics."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        max_llm_headings: int = _DEFAULT_MAX_LLM_HEADINGS,
        max_tokens: int = 1500,
    ) -> None:
        self.llm = llm
        self.max_llm_headings = max_llm_headings
        self.max_tokens = max_tokens

    def assign(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        """Return blocks with heading-block levels inferred (others untouched)."""
        heading_idx = [i for i, b in enumerate(blocks) if b.block_type == BlockType.HEADING]
        titles = [self._title(blocks[i]) for i in heading_idx]
        if not titles:
            return list(blocks)

        levels: list[int] | None = None
        if self.llm is not None and len(titles) <= self.max_llm_headings:
            levels = self._llm_levels(titles)
        if levels is None:
            levels = [self._heuristic_level(t) for t in titles]

        out = list(blocks)
        for idx, level in zip(heading_idx, levels, strict=True):
            block = out[idx]
            out[idx] = block.model_copy(
                update={"metadata": {**block.metadata, "heading_level": level}}
            )
        return out

    def _title(self, block: DocumentBlock) -> str:
        return block.text.lstrip("#").strip()

    def _heuristic_level(self, title: str) -> int:
        if _PART.match(title):
            return 1
        if _ITEM.match(title):
            return 2
        match = _NUMBERED.match(title)
        if match:
            return min(match.group(1).count(".") + 1, _MAX_LEVEL)
        return 3

    def _llm_levels(self, titles: list[str]) -> list[int] | None:
        listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        try:
            text = self.llm.generate(
                system_prompt=_SYSTEM, user_prompt=listing, max_tokens=self.max_tokens
            ).text
        except Exception as exc:
            logger.warning("Heading-hierarchy LLM call failed: %s", exc)
            return None

        match = re.search(r"\[[\s\d,]+\]", text)
        if not match:
            logger.warning("Heading-hierarchy LLM output had no JSON array; using heuristic")
            return None
        try:
            values = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(values, list) or len(values) != len(titles):
            return None
        if not all(isinstance(v, int) and v >= 1 for v in values):
            return None
        return [min(v, _MAX_LEVEL) for v in values]
