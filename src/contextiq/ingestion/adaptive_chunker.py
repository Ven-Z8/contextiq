"""Adaptive chunking: content-type-aware chunk strategy selection.

Based on arxiv:2603.25333 (Adaptive Chunking: Optimizing Chunking-Method
Selection for RAG), the key insight is: different content types have
fundamentally different retrieval needs and should be chunked differently.

One-size chunking fails enterprise documents because:
  - Financial tables split mid-row → corrupt number context
  - Explanatory prose chunked at word boundaries → loses sentence meaning
  - Risk factor paragraphs split mid-argument → retrieval returns half-arguments
  - Structured lists chunked monolithically → can't retrieve individual items

This module routes each DocumentBlock through a ContentProfile classifier
and applies the appropriate chunking strategy:

  FINANCIAL_TABLE  → keep-whole or row-boundary split (max 600 tokens)
  NARRATIVE_PARA   → sentence-window with overlap (256 tok, 64 tok stride)
  RISK_SECTION     → paragraph-boundary (keep whole paragraphs, max 512 tok)
  LIST_ITEMS       → item-level with parent heading prefix
  NUMERICAL_FACT   → anchor window (number + surrounding context)
  HEADING          → pass-through (attached to following content by hierarchy)
  GENERIC          → pass-through (default chunker handles overflow)
"""

from __future__ import annotations

import re
from enum import StrEnum

from contextiq.ingestion.models import BlockType, DocumentBlock

# Financial terminology that signals table/metric content
_FINANCIAL_KEYWORDS: frozenset[str] = frozenset({
    "revenue", "net sales", "operating income", "gross margin",
    "earnings per share", "eps", "cash flow", "total assets",
    "net income", "diluted", "fiscal", "quarter", "segment",
    "consolidated", "statements of operations", "balance sheet",
    "stockholders", "shareholders", "liabilities", "ebitda",
    "amortization", "depreciation", "capex", "free cash flow",
    "millions", "billions", "thousand", "percent", "increase",
    "decrease", "year ended", "three months", "nine months",
})

# Risk / compliance signal words
_RISK_KEYWORDS: frozenset[str] = frozenset({
    "risk", "risks", "uncertainty", "may adversely", "could adversely",
    "adverse", "regulation", "regulatory", "compliance", "litigation",
    "lawsuit", "legal", "tariff", "sanction", "exposure", "material",
    "uncertainty", "subject to", "contingent", "indemnification",
})

# Sentence boundary pattern (handles abbreviations imperfectly but well enough)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Dollar / large number pattern
_NUMBER_PATTERN = re.compile(r'\$[\d,]+|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?%')

# Common list item markers
_LIST_MARKER = re.compile(r'^\s*(?:•|[-–—*]|\d+[.)]\s+|[a-z][.)]\s+)', re.MULTILINE)


class ContentProfile(StrEnum):
    """Content type driving chunk strategy selection."""

    FINANCIAL_TABLE = "financial_table"
    NARRATIVE_PARA = "narrative_para"
    RISK_SECTION = "risk_section"
    LIST_ITEMS = "list_items"
    NUMERICAL_FACT = "numerical_fact"
    HEADING = "heading"
    GENERIC = "generic"


class AdaptiveChunker:
    """Route DocumentBlocks to content-appropriate chunking strategies.

    Usage:
        chunker = AdaptiveChunker()
        # Process a list of raw blocks from the parser:
        chunked_blocks = chunker.process_blocks(raw_blocks)
        # Each block now has content_profile in metadata and optimal chunk size.
    """

    def __init__(
        self,
        max_table_tokens: int = 600,
        max_prose_tokens: int = 256,
        prose_stride_tokens: int = 64,
        max_risk_tokens: int = 512,
        numerical_context_tokens: int = 100,
    ) -> None:
        self.max_table_tokens = max_table_tokens
        self.max_prose_tokens = max_prose_tokens
        self.prose_stride_tokens = prose_stride_tokens
        self.max_risk_tokens = max_risk_tokens
        self.numerical_context_tokens = numerical_context_tokens

    def classify(self, block: DocumentBlock) -> ContentProfile:
        """Classify a DocumentBlock into a ContentProfile."""
        text = block.text
        text_lower = text.lower()

        if block.block_type == BlockType.HEADING:
            return ContentProfile.HEADING

        if block.block_type == BlockType.TABLE:
            if any(kw in text_lower for kw in _FINANCIAL_KEYWORDS):
                return ContentProfile.FINANCIAL_TABLE
            return ContentProfile.GENERIC  # Non-financial table: default chunker handles

        # Text blocks: detect by content signals
        if any(kw in text_lower for kw in _RISK_KEYWORDS):
            return ContentProfile.RISK_SECTION

        # Check for list structure
        list_matches = _LIST_MARKER.findall(text)
        if len(list_matches) >= 3:
            return ContentProfile.LIST_ITEMS

        # Financial narrative (contains numbers + financial terms)
        has_numbers = bool(_NUMBER_PATTERN.search(text))
        has_financial = any(kw in text_lower for kw in _FINANCIAL_KEYWORDS)
        if has_numbers and has_financial:
            return ContentProfile.NUMERICAL_FACT

        # Long prose → sentence-window chunking
        word_count = len(text.split())
        if word_count > 80:
            return ContentProfile.NARRATIVE_PARA

        return ContentProfile.GENERIC

    def chunk(
        self,
        block: DocumentBlock,
        profile: ContentProfile,
    ) -> list[DocumentBlock]:
        """Apply the appropriate chunk strategy for the content profile."""

        if profile == ContentProfile.FINANCIAL_TABLE:
            return self._chunk_financial_table(block)

        if profile == ContentProfile.NARRATIVE_PARA:
            return self._chunk_sentence_window(block)

        if profile == ContentProfile.RISK_SECTION:
            return self._chunk_by_paragraph(block)

        if profile == ContentProfile.LIST_ITEMS:
            return self._chunk_list_items(block)

        if profile == ContentProfile.NUMERICAL_FACT:
            # Numerical facts: keep whole if small, else sentence-window
            if len(block.text.split()) <= self.numerical_context_tokens * 2:
                return [self._tag_profile(block, profile)]
            return self._chunk_sentence_window(block)

        # HEADING, GENERIC: no re-chunking, just tag the profile
        return [self._tag_profile(block, profile)]

    def process_blocks(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        """Classify and chunk a list of blocks.

        Each output block gets:
          - metadata["content_profile"]: ContentProfile value
          - metadata["chunk_strategy"]: which strategy was applied
          - block_id suffix ":adaptive-N" for sub-chunks
        """
        result: list[DocumentBlock] = []
        for block in blocks:
            profile = self.classify(block)
            chunks = self.chunk(block, profile)
            result.extend(chunks)
        return result

    # ── Chunking Strategies ───────────────────────────────────────────────────

    def _chunk_financial_table(self, block: DocumentBlock) -> list[DocumentBlock]:
        """Keep financial tables intact or split at row boundaries.

        Never splits mid-row — always includes header in every chunk.
        """
        lines = block.text.split("\n")
        table_lines = [l for l in lines if l.strip().startswith("|")]
        non_table = [l for l in lines if not l.strip().startswith("|")]

        if not table_lines:
            return [self._tag_profile(block, ContentProfile.FINANCIAL_TABLE)]

        # Find header (first row) and separator (second row: |---|---|)
        header_rows = []
        data_rows = []
        found_sep = False
        for line in table_lines:
            if not found_sep and re.match(r'\|[\s:-]+\|', line):
                found_sep = True
                header_rows.append(line)  # separator row stays with header
                continue
            if not found_sep:
                header_rows.append(line)
            else:
                data_rows.append(line)

        # Count tokens approximately (1 word ≈ 1 token for tables)
        header_text = "\n".join(header_rows)
        header_tokens = len(header_text.split())

        if header_tokens + len(data_rows) * 8 <= self.max_table_tokens:
            # Entire table fits → keep whole
            return [self._tag_profile(block, ContentProfile.FINANCIAL_TABLE,
                                      strategy="table_keep_whole")]

        # Split at row boundaries, repeat header in each chunk
        chunks: list[DocumentBlock] = []
        rows_per_chunk = max(
            1,
            (self.max_table_tokens - header_tokens) // max(1, len(data_rows) // 10 * 10)
        )
        rows_per_chunk = max(5, min(25, rows_per_chunk))  # 5-25 rows per chunk

        for chunk_idx, start in enumerate(range(0, len(data_rows), rows_per_chunk)):
            chunk_rows = data_rows[start : start + rows_per_chunk]
            chunk_text = header_text + "\n" + "\n".join(chunk_rows)
            if non_table:
                chunk_text = "\n".join(non_table) + "\n" + chunk_text
            chunks.append(self._make_sub_chunk(
                block=block,
                text=chunk_text,
                chunk_index=chunk_idx,
                profile=ContentProfile.FINANCIAL_TABLE,
                strategy="table_row_boundary",
                extra_meta={
                    "row_start": start + 1,
                    "row_end": start + len(chunk_rows),
                },
            ))

        return chunks or [self._tag_profile(block, ContentProfile.FINANCIAL_TABLE)]

    def _chunk_sentence_window(self, block: DocumentBlock) -> list[DocumentBlock]:
        """Sentence-boundary windowing with stride overlap.

        Preserves sentence integrity — never splits mid-sentence.
        Stride overlap ensures no context is lost between chunks.
        """
        sentences = _SENTENCE_SPLIT.split(block.text.strip())
        if not sentences:
            return [self._tag_profile(block, ContentProfile.NARRATIVE_PARA)]

        chunks: list[DocumentBlock] = []
        start_idx = 0
        chunk_idx = 0

        while start_idx < len(sentences):
            window: list[str] = []
            token_count = 0

            for sent in sentences[start_idx:]:
                sent_tokens = len(sent.split())
                if token_count + sent_tokens > self.max_prose_tokens and window:
                    break
                window.append(sent)
                token_count += sent_tokens

            if not window:
                window = [sentences[start_idx]]  # Always advance at least one sentence

            chunk_text = " ".join(window)
            chunks.append(self._make_sub_chunk(
                block=block,
                text=chunk_text,
                chunk_index=chunk_idx,
                profile=ContentProfile.NARRATIVE_PARA,
                strategy="sentence_window",
                extra_meta={
                    "sentence_start": start_idx + 1,
                    "sentence_end": start_idx + len(window),
                    "token_count": token_count,
                },
            ))

            # Advance by stride: overlap = window_sentences - stride_sentences
            stride_tokens = 0
            skip = 0
            for sent in window:
                stride_tokens += len(sent.split())
                skip += 1
                if stride_tokens >= self.prose_stride_tokens:
                    break
            start_idx += max(1, skip)
            chunk_idx += 1

        return chunks or [self._tag_profile(block, ContentProfile.NARRATIVE_PARA)]

    def _chunk_by_paragraph(self, block: DocumentBlock) -> list[DocumentBlock]:
        """Split at paragraph boundaries, keeping paragraphs intact.

        Paragraph = double newline. Merges short consecutive paragraphs
        until `max_risk_tokens` is reached.
        """
        paragraphs = re.split(r'\n\s*\n', block.text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [self._tag_profile(block, ContentProfile.RISK_SECTION)]

        chunks: list[DocumentBlock] = []
        current_paras: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para_tokens = len(para.split())
            if current_tokens + para_tokens > self.max_risk_tokens and current_paras:
                # Flush current chunk
                chunks.append(self._make_sub_chunk(
                    block=block,
                    text="\n\n".join(current_paras),
                    chunk_index=chunk_idx,
                    profile=ContentProfile.RISK_SECTION,
                    strategy="paragraph_boundary",
                ))
                current_paras = []
                current_tokens = 0
                chunk_idx += 1
            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            chunks.append(self._make_sub_chunk(
                block=block,
                text="\n\n".join(current_paras),
                chunk_index=chunk_idx,
                profile=ContentProfile.RISK_SECTION,
                strategy="paragraph_boundary",
            ))

        return chunks or [self._tag_profile(block, ContentProfile.RISK_SECTION)]

    def _chunk_list_items(self, block: DocumentBlock) -> list[DocumentBlock]:
        """Split list items individually, prefixing with the block's section path.

        Each list item becomes a standalone retrievable chunk. Items shorter than
        10 words are merged with the next item.
        """
        lines = block.text.split("\n")
        items: list[str] = []
        current_item: list[str] = []

        for line in lines:
            if _LIST_MARKER.match(line) and current_item:
                items.append(" ".join(current_item))
                current_item = [line.strip()]
            else:
                current_item.append(line.strip())
        if current_item:
            items.append(" ".join(current_item))

        # Merge very short items with next item
        merged: list[str] = []
        i = 0
        while i < len(items):
            if len(items[i].split()) < 10 and i + 1 < len(items):
                merged.append(items[i] + " " + items[i + 1])
                i += 2
            else:
                merged.append(items[i])
                i += 1

        if not merged:
            return [self._tag_profile(block, ContentProfile.LIST_ITEMS)]

        section_prefix = " > ".join(block.section_path) if block.section_path else ""
        chunks: list[DocumentBlock] = []
        for chunk_idx, item_text in enumerate(merged):
            full_text = f"{section_prefix}: {item_text}" if section_prefix else item_text
            chunks.append(self._make_sub_chunk(
                block=block,
                text=full_text,
                chunk_index=chunk_idx,
                profile=ContentProfile.LIST_ITEMS,
                strategy="list_item",
            ))

        return chunks or [self._tag_profile(block, ContentProfile.LIST_ITEMS)]

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _tag_profile(
        self,
        block: DocumentBlock,
        profile: ContentProfile,
        strategy: str = "keep_whole",
    ) -> DocumentBlock:
        """Return block with content_profile metadata added (no re-chunking)."""
        return block.model_copy(update={
            "metadata": {
                **block.metadata,
                "content_profile": str(profile),
                "chunk_strategy": strategy,
            }
        })

    def _make_sub_chunk(
        self,
        block: DocumentBlock,
        text: str,
        chunk_index: int,
        profile: ContentProfile,
        strategy: str,
        extra_meta: dict | None = None,
    ) -> DocumentBlock:
        """Create a sub-chunk with updated block_id and metadata."""
        meta = {
            **block.metadata,
            "content_profile": str(profile),
            "chunk_strategy": strategy,
            "parent_block_id": block.block_id,
            "chunk_index": chunk_index,
        }
        if extra_meta:
            meta.update(extra_meta)

        return block.model_copy(update={
            "block_id": f"{block.block_id}:adaptive-{chunk_index}",
            "text": text,
            "metadata": meta,
        })
