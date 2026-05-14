from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.candidates import CandidateGenerator
from contextiq.retrieval.expansion import SectionExpander
from contextiq.retrieval.pipeline import RetrievalPipeline
from contextiq.retrieval.query import QueryAnalyzer
from contextiq.retrieval.ranker import CandidateRanker


def test_candidate_generator_combines_vector_lexical_section_and_financial_anchors() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:heading",
            source_path="sample.md",
            section_path=["Products and Services Performance"],
            block_type=BlockType.HEADING,
            text="## Products and Services Performance",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:table",
            source_path="sample.md",
            section_path=["Products and Services Performance"],
            block_type=BlockType.TABLE,
            text="| Total net sales | 2025 | 2024 | 2023 |",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:other",
            source_path="sample.md",
            section_path=["Other"],
            text="Generic prose.",
        ),
    ]

    generator = CandidateGenerator(
        blocks_provider=lambda: blocks,
        vector_search=lambda _query, _limit: [blocks[1]],
        analyzer=QueryAnalyzer(),
        ranker=CandidateRanker(),
    )

    candidates = generator.generate(
        "What is the Products and Services Performance in 2025?",
        limit=10,
    )

    assert [block.block_id for block in candidates] == [
        "doc:table",
        "doc:heading",
    ]


def test_candidate_generator_keeps_each_source_candidate_budget() -> None:
    vector_blocks = [
        DocumentBlock(
            document_id="vector",
            block_id=f"vector:{index}",
            source_path="vector.md",
            text="Financial filing context.",
        )
        for index in range(3)
    ]
    lexical_match = DocumentBlock(
        document_id="contract",
        block_id="contract:termination",
        source_path="contract.md",
        section_path=["Termination"],
        text="Either party may terminate the agreement on written notice.",
    )
    blocks = [*vector_blocks, lexical_match]
    generator = CandidateGenerator(
        blocks_provider=lambda: blocks,
        vector_search=lambda _query, _limit: vector_blocks,
    )

    candidates = generator.generate("How can either party terminate?", limit=3)

    assert candidates[-1].block_id == "contract:termination"


def test_section_expander_adds_same_section_heading_and_neighbors() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:heading",
            source_path="sample.md",
            section_path=["Confidentiality"],
            block_type=BlockType.HEADING,
            text="## Confidentiality",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:evidence",
            source_path="sample.md",
            section_path=["Confidentiality"],
            text="Each party must protect confidential information.",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:neighbor",
            source_path="sample.md",
            section_path=["Confidentiality"],
            text="Disclosure is limited to advisors.",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:next",
            source_path="sample.md",
            section_path=["Termination"],
            block_type=BlockType.HEADING,
            text="## Termination",
        ),
    ]

    expanded = SectionExpander(blocks_provider=lambda: blocks).expand(
        candidates=[blocks[1]],
        limit=5,
        query="What confidentiality obligations do the parties have?",
    )

    assert [block.block_id for block in expanded] == [
        "doc:heading",
        "doc:evidence",
        "doc:neighbor",
    ]


def test_retrieval_pipeline_generates_expands_reranks_and_filters() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:heading",
            source_path="sample.md",
            section_path=["Products and Services Performance"],
            block_type=BlockType.HEADING,
            text="## Products and Services Performance",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:table",
            source_path="sample.md",
            section_path=["Products and Services Performance"],
            block_type=BlockType.TABLE,
            text=(
                "| Category | 2025 | 2024 | 2023 |\n"
                "| iPhone | $209,586 | $201,183 | $200,583 |\n"
                "| Services | $109,158 | $96,169 | $85,200 |\n"
                "| Total net sales | $416,161 | $391,035 | $383,285 |"
            ),
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:risk",
            source_path="sample.md",
            section_path=["Risk Factors"],
            text=(
                "Products and services performance may be adversely affected by market "
                "risk during 2025."
            ),
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: [blocks[2]],
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search(
        "What is the Products and Services Performance in 2025?",
        limit=3,
    )

    result_ids = [block.block_id for block in results]
    assert result_ids[0] == "doc:table"
    assert "doc:risk" not in result_ids


def test_retrieval_pipeline_exposes_source_stage_trace() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:heading",
            source_path="sample.md",
            section_path=["Revenue Chart"],
            block_type=BlockType.HEADING,
            text="## Revenue Chart",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:figure",
            source_path="sample.md",
            section_path=["Revenue Chart"],
            block_type=BlockType.FIGURE,
            text=(
                "Figure: Revenue by segment chart. "
                "Visual description: Americas Europe Services."
            ),
            metadata={
                "visual_kind": "figure",
                "caption": "Revenue by segment chart",
                "visual_description": "Americas Europe Services.",
            },
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: [],
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    hits = pipeline.search_with_trace("Find the revenue by segment chart", limit=2)

    figure_hit = next(hit for hit in hits if hit.block.block_id == "doc:figure")
    assert "lexical" in figure_hit.stages
    assert figure_hit.rank == 1
    assert figure_hit.score > 0
    assert "figure" in figure_hit.reason.lower()


def test_visual_intent_query_prioritizes_visual_evidence() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:text",
            source_path="report.pdf",
            section_path=["Architecture"],
            text=(
                "The architecture services section describes data access, delivery, "
                "and governance capabilities in prose."
            ),
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:figure",
            source_path="report.pdf",
            section_path=["Architecture"],
            block_type=BlockType.FIGURE,
            text=(
                "Figure: Reference architecture service diagram.\n"
                "Visual description: A flow chart showing data services and computing "
                "services."
            ),
            metadata={
                "visual_kind": "figure",
                "caption": "Reference architecture service diagram.",
                "visual_class": "flow_chart",
                "visual_description": "A flow chart showing data services.",
            },
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: [blocks[0]],
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    hits = pipeline.search_with_trace(
        "Find the diagram about data and computing architecture services.",
        limit=2,
    )

    assert hits[0].block.block_id == "doc:figure"
    assert "visual" in hits[0].stages


def test_visual_intent_ignores_empty_decorative_figures() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:empty-figure",
            source_path="report.pdf",
            block_type=BlockType.FIGURE,
            text="<figure>",
            metadata={"visual_kind": "figure"},
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:described-figure",
            source_path="report.pdf",
            block_type=BlockType.FIGURE,
            text="Figure: Architecture diagram.",
            metadata={
                "visual_kind": "figure",
                "caption": "Architecture diagram.",
                "visual_class": "flow_chart",
            },
        ),
    ]
    generator = CandidateGenerator(
        blocks_provider=lambda: blocks,
        vector_search=lambda _query, _limit: [],
    )

    candidates = generator.generate_with_trace("Find the architecture diagram.", limit=5)

    assert [candidate.block.block_id for candidate in candidates] == [
        "doc:described-figure"
    ]


def test_query_analyzer_detects_visual_intent() -> None:
    analysis = QueryAnalyzer().analyze(
        "Find the chart or diagram about architecture services."
    )

    assert analysis.has_visual_intent


def test_retrieval_pipeline_filters_to_exact_structured_code_matches() -> None:
    blocks = [
        DocumentBlock(
            document_id="apple",
            block_id="apple:2022-plan",
            source_path="apple-2025-10k.pdf",
            section_path=["2022 Employee Stock Plan"],
            text="The 2022 Plan authorizes RSUs and stock options.",
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:uc-t-202",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Use Cases"],
            block_type=BlockType.TABLE,
            text=(
                "| UC ID | Use Cases |\n"
                "| UC-T-202 L | Transportation of large cargo from Earth to the lunar surface |"
            ),
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:uc-t-201",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Use Cases"],
            block_type=BlockType.TABLE,
            text="| UC ID | Use Cases |\n| UC-T-201 L | Transportation of small cargo |",
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: [blocks[0]],
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search("What is UC ID UC-T-202 L ?", limit=5)

    assert [block.block_id for block in results] == ["nasa:uc-t-202"]


def test_retrieval_pipeline_filters_to_exact_asset_mapping_matches() -> None:
    blocks = [
        DocumentBlock(
            document_id="apple",
            block_id="apple:plan",
            source_path="apple-2025-10k.pdf",
            section_path=["2022 Employee Stock Plan"],
            text="The plan supports employees with awards and functions.",
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:function-list",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Functions"],
            block_type=BlockType.TABLE,
            text=(
                "| FN ID | Functions |\n"
                "| FN-T-101 L | Transport crew from Earth to cislunar space |"
            ),
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:orion-fe",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Asset-Function Table"],
            block_type=BlockType.TABLE,
            text=(
                "| Asset ID | Assets | FN ID | Functions |\n"
                "| EM-003-FE | Orion | FN-U-206 L | Provide intravehicular "
                "utilization accommodation |"
            ),
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:orion-hlr",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Asset-Function Table"],
            block_type=BlockType.TABLE,
            text=(
                "| Asset ID | Assets | FN ID | Functions |\n"
                "| EM-003-HLR | Orion | FN-T-101 L | Transport crew from "
                "Earth to cislunar space |\n"
                "|  |  | FN-L-302 L | Manage waste from habitable asset(s) in cislunar space |"
            ),
        ),
        DocumentBlock(
            document_id="nasa",
            block_id="nasa:gateway-hlr",
            source_path="2024-lunar-objective-decomposition.xlsx",
            section_path=["Asset-Function Table"],
            block_type=BlockType.TABLE,
            text=(
                "| Asset ID | Assets | FN ID | Functions |\n"
                "| EM-004-HLR | Gateway | FN-H-103 L | Enable a pressurized environment |"
            ),
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: [blocks[0], blocks[1], blocks[2]],
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search("What HLR functions support for Orion?", limit=5)

    assert [block.block_id for block in results] == ["nasa:orion-hlr"]


def test_retrieval_pipeline_filters_contract_queries_to_contract_evidence() -> None:
    blocks = [
        DocumentBlock(
            document_id="apple",
            block_id="apple:agreements",
            source_path="apple-2025-10k.pdf",
            section_path=["Legal Proceedings"],
            text="The Company enters into agreements and may settle litigation.",
        ),
        DocumentBlock(
            document_id="contract",
            block_id="contract:termination",
            source_path="sample-contract.md",
            section_path=["Termination"],
            text=(
                "Either party may terminate for material breach if the breach remains "
                "uncured for thirty days after written notice."
            ),
        ),
        DocumentBlock(
            document_id="contract",
            block_id="contract:renewal",
            source_path="sample-contract.md",
            section_path=["Term and Renewal"],
            text="The agreement renews unless either party gives sixty days notice.",
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: blocks,
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search("How can either party terminate the agreement?", limit=5)

    assert [block.block_id for block in results] == [
        "contract:termination",
        "contract:renewal",
    ]


def test_retrieval_pipeline_prefers_named_topic_sources_over_generic_matches() -> None:
    blocks = [
        DocumentBlock(
            document_id="azure-agent-patterns",
            block_id="azure:raw",
            source_path="agent-factory.md",
            section_path=["Raw Extract"],
            text=(
                "Retrieval augmented generation helps teams find documents and "
                "agents can use tools to store business records."
            ),
        ),
        DocumentBlock(
            document_id="qdrant-retrieval-architecture",
            block_id="qdrant:payload",
            source_path="qdrant-retrieval-architecture-deep-dive.md",
            section_path=["Payload Schema For Complex Documents"],
            text=(
                "Qdrant payload fields should include document_id, source_path, "
                "page, section_path, block_type, table_id, figure_id, and parser."
            ),
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: blocks,
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search(
        "What payload fields should ContextIQ store in Qdrant for complex document retrieval?",
        limit=2,
    )

    assert results[0].block_id == "qdrant:payload"


def test_retrieval_pipeline_filters_financial_queries_to_named_company_source() -> None:
    blocks = [
        DocumentBlock(
            document_id="apple-2025-10k",
            block_id="apple:income",
            source_path="apple-2025-10k.pdf",
            block_type=BlockType.TABLE,
            text=(
                "| Revenue | Gross margin | Operating income | Net income |\n"
                "| $416,161 | $195,201 | $133,050 | $112,010 |"
            ),
        ),
        DocumentBlock(
            document_id="MSFT_FY25q4_10K",
            block_id="msft:income",
            source_path="MSFT_FY25q4_10K.docx",
            block_type=BlockType.TABLE,
            text=(
                "| Revenue | Gross margin | Operating income | Net income |\n"
                "| $281,724 | $193,893 | $128,528 | $101,832 |"
            ),
        ),
        DocumentBlock(
            document_id="MSFT_FY25q4_10K",
            block_id="msft:segment-note",
            source_path="MSFT_FY25q4_10K.docx",
            text="Microsoft changed segment reporting in fiscal year 2025.",
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: blocks,
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search(
        (
            "What was Microsoft revenue, gross margin, operating income, "
            "and net income in fiscal 2025?"
        ),
        limit=2,
    )

    assert [block.block_id for block in results] == ["msft:income"]


def test_retrieval_pipeline_filters_product_service_query_to_category_evidence() -> None:
    blocks = [
        DocumentBlock(
            document_id="apple-2025-10k",
            block_id="apple:products-services",
            source_path="apple-2025-10k.pdf",
            section_path=["Products and Services Performance"],
            block_type=BlockType.TABLE,
            text=(
                "| iPhone | 2025 | 2024 | 2023 |\n"
                "| Services | 109,158 | 96,169 | 85,200 |\n"
                "| Total net sales | 416,161 | 391,035 | 383,285 |"
            ),
        ),
        DocumentBlock(
            document_id="nvidia-annual-report",
            block_id="nvidia:income",
            source_path="NVIDIA-2025-Annual-Report.pdf",
            block_type=BlockType.TABLE,
            text=(
                "| Revenue | Gross margin | Operating income | Net income |\n"
                "| 130,497 | 75.0% | 81,453 | 72,880 |"
            ),
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: blocks,
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search(
        "What is the Products and Services Performance in 2025?",
        limit=2,
    )

    assert [block.block_id for block in results] == ["apple:products-services"]


def test_retrieval_pipeline_filters_named_report_title_and_prefers_recommendations() -> None:
    blocks = [
        DocumentBlock(
            document_id="data-and-computing-architecture-study-final-report",
            block_id="study:title-date",
            source_path="data-and-computing-architecture-study-final-report-aug-2024.pdf",
            text="August 2024",
        ),
        DocumentBlock(
            document_id="data-and-computing-architecture-study-final-report",
            block_id="study:recommendation",
            source_path="data-and-computing-architecture-study-final-report-aug-2024.pdf",
            text=(
                "Recommendation 2.1: OCSDO should develop a single interoperable "
                "hybrid cloud and HEC data and computing architecture."
            ),
        ),
        DocumentBlock(
            document_id="2024-lunar-objective-decomposition",
            block_id="lunar:data",
            source_path="2024-lunar-objective-decomposition.xlsx",
            block_type=BlockType.TABLE,
            text=(
                "| Functions | Use Cases |\n"
                "| Provide communications and data exchange between lunar surface and Earth |"
            ),
        ),
    ]
    ranker = CandidateRanker()
    pipeline = RetrievalPipeline(
        generator=CandidateGenerator(
            blocks_provider=lambda: blocks,
            vector_search=lambda _query, _limit: blocks,
            ranker=ranker,
        ),
        expander=SectionExpander(blocks_provider=lambda: blocks, ranker=ranker),
        ranker=ranker,
    )

    results = pipeline.search(
        "What did the NASA Data and Computing Architecture Study recommend?",
        limit=3,
    )

    assert [block.block_id for block in results] == ["study:recommendation"]
