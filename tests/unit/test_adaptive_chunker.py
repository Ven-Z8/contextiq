"""Unit tests for AdaptiveChunker — content-type-aware chunk strategies.

All tests use synthetic DocumentBlock instances (no corpus needed).
Tests verify classification accuracy and chunk strategy correctness.
"""

from __future__ import annotations

from contextiq.ingestion.adaptive_chunker import (
    AdaptiveChunker,
    ContentProfile,
)
from contextiq.ingestion.models import BlockType, DocumentBlock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_block(
    text: str,
    block_type: BlockType = BlockType.TEXT,
    section_path: list[str] | None = None,
    block_id: str = "test-doc:0",
) -> DocumentBlock:
    return DocumentBlock(
        document_id="test-doc",
        block_id=block_id,
        source_path="test.pdf",
        block_type=block_type,
        section_path=section_path or [],
        text=text,
    )


FINANCIAL_TABLE_TEXT = """\
| Segment | 2025 | 2024 | Change |
| --- | --- | --- | --- |
| Products | 294,866 | 278,108 | 6% |
| Services | 109,158 | 96,169 | 13% |
| Total net sales | 416,161 | 383,285 | 9% |
| Gross margin | 195,201 | 180,683 | 8% |
"""

NARRATIVE_PARA_TEXT = (
    "The increase in net sales during 2025 was primarily driven by higher unit sales "
    "across iPhone, Mac, and Services. Services growth was particularly strong due to "
    "increased App Store activity and advertising revenue. The company continued to "
    "invest in research and development, with R&D spending up 12% year over year. "
    "Management believes these investments will drive long-term sustainable growth. "
    "The geographic distribution of sales remained consistent with prior years."
)

RISK_SECTION_TEXT = (
    "We are subject to complex and changing laws and regulations governing privacy, "
    "data protection, and security. Our business may be adversely affected by any "
    "failure to comply with applicable requirements. Regulatory uncertainty could "
    "result in material adverse effects on our operations. Tariff exposure from "
    "imported components could adversely impact our cost structure."
)

LIST_TEXT = """\
The following risk factors may adversely affect our business:
• Supply chain disruptions from geopolitical events
• Increased competition in key product categories
• Foreign exchange rate fluctuations impacting revenue
• Changes in consumer preferences and technology trends
"""


# ---------------------------------------------------------------------------
# Classification Tests
# ---------------------------------------------------------------------------

class TestContentProfileClassification:
    """Test that blocks are classified into the correct ContentProfile."""

    def setup_method(self):
        self.chunker = AdaptiveChunker()

    def test_heading_block_classified_as_heading(self):
        block = _make_block("Financial Statements", block_type=BlockType.HEADING)
        assert self.chunker.classify(block) == ContentProfile.HEADING

    def test_financial_table_classified_correctly(self):
        block = _make_block(FINANCIAL_TABLE_TEXT, block_type=BlockType.TABLE)
        assert self.chunker.classify(block) == ContentProfile.FINANCIAL_TABLE

    def test_non_financial_table_falls_back_to_generic(self):
        block = _make_block(
            "| Name | Department | Location |\n| --- | --- | --- |\n| Alice | Eng | NYC |",
            block_type=BlockType.TABLE,
        )
        assert self.chunker.classify(block) == ContentProfile.GENERIC

    def test_risk_section_classified_correctly(self):
        block = _make_block(RISK_SECTION_TEXT)
        assert self.chunker.classify(block) == ContentProfile.RISK_SECTION

    def test_long_prose_classified_as_narrative(self):
        block = _make_block(NARRATIVE_PARA_TEXT)
        assert self.chunker.classify(block) == ContentProfile.NARRATIVE_PARA

    def test_list_items_classified_correctly(self):
        block = _make_block(LIST_TEXT)
        assert self.chunker.classify(block) == ContentProfile.LIST_ITEMS

    def test_short_text_classified_as_generic(self):
        block = _make_block("This is a short caption.")
        assert self.chunker.classify(block) == ContentProfile.GENERIC

    def test_numerical_fact_with_dollar_amounts(self):
        block = _make_block(
            "Apple reported $416,161 million in total net sales for fiscal 2025, "
            "representing an increase compared to prior year revenue figures."
        )
        profile = self.chunker.classify(block)
        assert profile in (ContentProfile.NUMERICAL_FACT, ContentProfile.NARRATIVE_PARA)


# ---------------------------------------------------------------------------
# Chunking Strategy Tests
# ---------------------------------------------------------------------------

class TestChunkingStrategies:
    """Test that each chunking strategy produces correct output."""

    def setup_method(self):
        self.chunker = AdaptiveChunker(
            max_table_tokens=100,   # Small limit to force splitting in tests
            max_prose_tokens=50,
            prose_stride_tokens=20,
            max_risk_tokens=80,
        )

    def test_financial_table_small_kept_whole(self):
        block = _make_block(FINANCIAL_TABLE_TEXT, block_type=BlockType.TABLE)
        chunker = AdaptiveChunker()  # Default limits
        chunks = chunker.chunk(block, ContentProfile.FINANCIAL_TABLE)
        # Small table should stay as one chunk
        assert len(chunks) >= 1
        assert all(b.document_id == block.document_id for b in chunks)

    def test_financial_table_chunks_always_have_profile(self):
        block = _make_block(FINANCIAL_TABLE_TEXT, block_type=BlockType.TABLE)
        chunks = self.chunker.chunk(block, ContentProfile.FINANCIAL_TABLE)
        for chunk in chunks:
            assert chunk.metadata.get("content_profile") == "financial_table"

    def test_narrative_para_produces_overlapping_windows(self):
        block = _make_block(NARRATIVE_PARA_TEXT)
        chunker = AdaptiveChunker(max_prose_tokens=30, prose_stride_tokens=10)
        chunks = chunker.chunk(block, ContentProfile.NARRATIVE_PARA)
        # Should produce multiple chunks for long text
        assert len(chunks) >= 2
        # All chunks have the right profile
        for chunk in chunks:
            assert chunk.metadata.get("content_profile") == "narrative_para"
            assert chunk.metadata.get("chunk_strategy") == "sentence_window"

    def test_narrative_para_chunk_ids_are_unique(self):
        block = _make_block(NARRATIVE_PARA_TEXT)
        chunker = AdaptiveChunker(max_prose_tokens=30, prose_stride_tokens=10)
        chunks = chunker.chunk(block, ContentProfile.NARRATIVE_PARA)
        block_ids = [c.block_id for c in chunks]
        assert len(block_ids) == len(set(block_ids)), "Duplicate block IDs in chunks"

    def test_risk_section_splits_at_paragraph_boundary(self):
        # Text with clear paragraph boundaries
        multi_para_text = (
            "We face significant regulatory risks from data protection laws.\n\n"
            "Changes in tariff policy may adversely affect our cost structure.\n\n"
            "Litigation exposure from intellectual property claims could be material.\n\n"
            "Climate-related risks may impact our supply chain and operations."
        )
        block = _make_block(multi_para_text)
        chunks = self.chunker.chunk(block, ContentProfile.RISK_SECTION)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("content_profile") == "risk_section"
            assert chunk.metadata.get("chunk_strategy") == "paragraph_boundary"

    def test_list_items_each_item_is_a_chunk(self):
        block = _make_block(LIST_TEXT, section_path=["Risk Factors"])
        chunks = self.chunker.chunk(block, ContentProfile.LIST_ITEMS)
        assert len(chunks) >= 3  # At least 3 distinct items
        for chunk in chunks:
            assert chunk.metadata.get("content_profile") == "list_items"

    def test_heading_is_passed_through_unchanged(self):
        block = _make_block("Revenue by Segment", block_type=BlockType.HEADING)
        chunks = self.chunker.chunk(block, ContentProfile.HEADING)
        assert len(chunks) == 1
        assert chunks[0].text == block.text
        assert chunks[0].metadata.get("content_profile") == "heading"

    def test_generic_is_passed_through(self):
        block = _make_block("Short generic text.")
        chunks = self.chunker.chunk(block, ContentProfile.GENERIC)
        assert len(chunks) == 1
        assert chunks[0].text == block.text


# ---------------------------------------------------------------------------
# process_blocks Integration Tests
# ---------------------------------------------------------------------------

class TestProcessBlocks:
    """Test end-to-end process_blocks() on mixed block lists."""

    def setup_method(self):
        self.chunker = AdaptiveChunker()

    def test_process_blocks_returns_tagged_blocks(self):
        blocks = [
            _make_block("Revenue by Segment", BlockType.HEADING, block_id="doc:0"),
            _make_block(FINANCIAL_TABLE_TEXT, BlockType.TABLE, block_id="doc:1"),
            _make_block(NARRATIVE_PARA_TEXT, block_id="doc:2"),
        ]
        result = self.chunker.process_blocks(blocks)
        assert len(result) >= 3  # May produce more due to chunking
        for block in result:
            assert "content_profile" in block.metadata, (
                f"Missing content_profile on {block.block_id}"
            )

    def test_process_blocks_all_block_ids_unique(self):
        blocks = [
            _make_block(NARRATIVE_PARA_TEXT + " " + str(i), block_id=f"doc:{i}")
            for i in range(5)
        ]
        result = self.chunker.process_blocks(blocks)
        block_ids = [b.block_id for b in result]
        assert len(block_ids) == len(set(block_ids)), "Duplicate block IDs after process_blocks"

    def test_process_blocks_preserves_document_id(self):
        blocks = [
            _make_block(FINANCIAL_TABLE_TEXT, BlockType.TABLE, block_id="apple-2025:0"),
        ]
        result = self.chunker.process_blocks(blocks)
        for block in result:
            assert block.document_id == "test-doc"

    def test_process_blocks_empty_input(self):
        result = self.chunker.process_blocks([])
        assert result == []

    def test_process_blocks_single_heading(self):
        blocks = [_make_block("Introduction", BlockType.HEADING, block_id="doc:0")]
        result = self.chunker.process_blocks(blocks)
        assert len(result) == 1
        assert result[0].metadata["content_profile"] == "heading"
