"""Candidate ranking strategies for retrieval."""

from __future__ import annotations

from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.profile import RetrievalProfile
from contextiq.retrieval.query import QueryAnalysis, QueryAnalyzer, QueryIntent


class CandidateRanker:
    """Score and filter retrieval candidates using explicit query analysis."""

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        profile: RetrievalProfile | None = None,
    ) -> None:
        self.analyzer = analyzer or QueryAnalyzer(profile=profile)
        self.profile = self.analyzer.profile

    def rerank(self, query: str, candidates: list[DocumentBlock]) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        unique = self._dedupe(candidates)
        return sorted(unique, key=lambda block: self.score(analysis, block), reverse=True)

    def apply_intent_precision(
        self, query: str, candidates: list[DocumentBlock]
    ) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        if analysis.has_structured_codes:
            precise = [
                block
                for block in candidates
                if self.matches_structured_code(analysis, block)
            ]
            return precise if precise else candidates
        if analysis.has_asset_mapping_query:
            precise = [
                block
                for block in candidates
                if self.matches_asset_mapping(analysis, block)
            ]
            return precise if precise else candidates
        source_precise = self.source_precision_candidates(analysis, candidates)
        if source_precise:
            candidates = source_precise
        if self.is_recommendation_query(analysis):
            recommendation_candidates = [
                block for block in candidates if self.recommendation_bonus(block) > 0
            ]
            if recommendation_candidates:
                candidates = recommendation_candidates
        if analysis.intent == QueryIntent.CONTRACT:
            contract_named = [
                block for block in candidates if self.is_contract_named_source(block)
            ]
            if contract_named:
                return contract_named
            precise = [block for block in candidates if self.is_contract_evidence(block)]
            return precise if precise else candidates
        if analysis.intent != QueryIntent.FINANCIAL_PERFORMANCE:
            return candidates

        if {"products", "services"} <= analysis.raw_terms:
            product_service_precise = [
                block
                for block in candidates
                if self.is_product_services_performance_evidence(block)
            ]
            if product_service_precise:
                return product_service_precise

        precise = [
            block
            for block in candidates
            if self.is_financial_performance_evidence(block)
        ]
        return precise if precise else candidates

    def heading_expansion_window(
        self, query: str | None, heading: DocumentBlock
    ) -> int:
        if query:
            analysis = self.analyzer.analyze(query)
            if (
                analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE
                and self.is_financial_anchor_heading(heading)
            ):
                return 14
            if self.is_query_matching_heading(query, heading):
                return 12
        return 3

    def is_query_matching_heading(self, query: str, heading: DocumentBlock) -> bool:
        if heading.block_type.value != "heading":
            return False
        analysis = self.analyzer.analyze(query)
        section_text = self.block_text_with_sections(heading)
        return any(term in section_text for term in analysis.terms)

    def score(self, analysis: QueryAnalysis, block: DocumentBlock) -> float:
        text = " ".join([block.text, *block.section_path]).lower()
        section_text = " ".join(block.section_path).lower()
        score = 0.0
        score += sum(3.0 for term in analysis.terms if term in text)
        score += sum(5.0 for term in analysis.terms if term in section_text)
        score += min(len(block.text) / 400, 2.0)
        if analysis.has_structured_codes:
            score += self.structured_code_bonus(analysis, block)
        if analysis.has_asset_mapping_query:
            score += self.asset_mapping_bonus(analysis, block)
        if block.block_type.value == "text":
            score += 2.0
        if block.block_type.value == "heading":
            score -= 1.5
        if analysis.has_visual_intent:
            score += self.visual_evidence_bonus(analysis, block)
        if block.block_type.value == "table" and "table" not in analysis.terms:
            score -= 1.0
        if self.is_generic_heading(block):
            score -= 4.0
        if self.is_low_information_frontmatter(block):
            score -= 30.0
        if self.is_recommendation_query(analysis):
            score += self.recommendation_bonus(block)
        if not analysis.mentions_gross_margin and self.is_gross_margin_block(block):
            score -= 18.0
        if analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE:
            score += self.financial_performance_bonus(analysis, block)
            score += self.financial_metric_table_bonus(block)
        elif analysis.intent == QueryIntent.CONTRACT:
            score += self.contract_bonus(block)
        elif not analysis.mentions_risk and self.is_risk_context(block):
            score -= 4.0
        score += self.topic_affinity_bonus(analysis, block)
        return score

    def topic_affinity_bonus(self, analysis: QueryAnalysis, block: DocumentBlock) -> float:
        """Prefer blocks from sources that match explicit named topics in the query."""

        anchors = self.topic_anchors(analysis)
        if not anchors:
            return 0.0

        text = self.block_text_with_sections(block)
        source_text = f"{block.document_id} {block.source_path}".lower()
        matched = {anchor for anchor in anchors if anchor in text or anchor in source_text}
        if matched:
            source_matches = sum(1 for anchor in matched if anchor in source_text)
            return 10.0 + (4.0 * source_matches)
        return -18.0

    def visual_evidence_bonus(self, analysis: QueryAnalysis, block: DocumentBlock) -> float:
        if not analysis.has_visual_intent:
            return 0.0

        metadata = block.metadata
        visual_kind = str(metadata.get("visual_kind") or "").lower()
        visual_class = str(metadata.get("visual_class") or "").lower()
        text = self.block_text_with_sections(block)
        has_visual_evidence = any(
            metadata.get(key)
            for key in (
                "caption",
                "visual_description",
                "visual_class",
                "visual_classes",
                "image_path",
            )
        )

        if block.block_type.value not in {"figure", "table"} and not visual_kind:
            return 0.0
        if not has_visual_evidence:
            return 0.0

        bonus = 18.0
        if block.block_type.value == "figure" or visual_kind == "figure":
            bonus += 16.0
        if visual_class:
            bonus += 8.0
        if metadata.get("visual_description"):
            bonus += 10.0
        if metadata.get("caption"):
            bonus += 6.0
        visual_matches = {
            term
            for term in analysis.visual_terms
            if term in text or term in visual_class
        }
        bonus += 8.0 * min(len(visual_matches), 3)
        return bonus

    def topic_anchors(self, analysis: QueryAnalysis) -> set[str]:
        return analysis.raw_terms & self.profile.topic_anchors

    def source_precision_candidates(
        self,
        analysis: QueryAnalysis,
        candidates: list[DocumentBlock],
    ) -> list[DocumentBlock]:
        """Constrain broad finance questions when the user names a source entity."""

        aliases = self.source_aliases(analysis)
        if not aliases:
            return []
        precise = [
            block
            for block in candidates
            if any(alias in self.source_identity_text(block) for alias in aliases)
        ]
        return precise

    def source_aliases(self, analysis: QueryAnalysis) -> set[str]:
        aliases: set[str] = set()
        for term, term_aliases in self.profile.source_aliases.items():
            if term in analysis.raw_terms:
                aliases.update(term_aliases)
        for title_alias in self.profile.document_title_aliases:
            if title_alias.terms <= analysis.raw_terms:
                aliases.update(title_alias.aliases)
        return aliases

    def source_identity_text(self, block: DocumentBlock) -> str:
        return f"{block.document_id} {block.source_path}".lower()

    def structured_code_bonus(
        self, analysis: QueryAnalysis, block: DocumentBlock
    ) -> float:
        if not self.matches_structured_code(analysis, block):
            return -50.0

        text = self.block_text_with_sections(block)
        bonus = 40.0
        if block.block_type.value == "table":
            bonus += 8.0
        if self.has_structured_source_marker(block):
            bonus += 10.0
        if any(code.endswith(" l") and code in text for code in analysis.structured_codes):
            bonus += 8.0
        return bonus

    def matches_structured_code(
        self, analysis: QueryAnalysis,
        block: DocumentBlock,
    ) -> bool:
        text = self.block_text_with_sections(block)
        return any(code in text for code in analysis.structured_codes)

    def asset_mapping_bonus(
        self, analysis: QueryAnalysis, block: DocumentBlock
    ) -> float:
        if not self.matches_asset_mapping(analysis, block):
            return -60.0

        bonus = 45.0
        section_text = " ".join(block.section_path).lower()
        if block.block_type.value == "table":
            bonus += 10.0
        if "asset-function" in section_text:
            bonus += 12.0
        if self.has_structured_source_marker(block):
            bonus += 8.0
        return bonus

    def matches_asset_mapping(
        self, analysis: QueryAnalysis,
        block: DocumentBlock,
    ) -> bool:
        if not analysis.has_asset_mapping_query:
            return False
        if block.block_type.value != "table":
            return False
        return self._has_asset_mode_row(
            text=block.text.lower(),
            asset_terms=analysis.asset_terms,
            asset_modes=analysis.asset_modes,
        )

    def _has_asset_mode_row(
        self,
        text: str,
        asset_terms: set[str],
        asset_modes: set[str],
    ) -> bool:
        for line in text.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            row_text = " ".join(cells)
            if not any(asset in row_text for asset in asset_terms):
                continue
            if any(f"-{mode}" in row_text for mode in asset_modes):
                return True
        return False

    def financial_performance_bonus(
        self, analysis: QueryAnalysis, block: DocumentBlock
    ) -> float:
        text = self.block_text_with_sections(block)
        bonus = 0.0
        if block.block_type.value == "table":
            bonus += 7.0
        if self.has_financial_anchor_text(text):
            bonus += 8.0
        if "net sales" in text:
            bonus += 5.0
        if "gross margin" in text and analysis.mentions_gross_margin:
            bonus += 4.0
        if "dollars in millions" in text:
            bonus += 3.0
        if {"2025", "2024", "2023"}.issubset(set(text.split())):
            bonus += 2.0
        if self.is_risk_context(block):
            bonus -= 8.0
        return bonus

    def financial_metric_table_bonus(self, block: DocumentBlock) -> float:
        if block.block_type.value != "table":
            return 0.0
        text = self.block_text_with_sections(block)
        metric_count = self.financial_metric_count(text)
        if metric_count < 3:
            return 0.0
        return 6.0 + (4.0 * metric_count)

    def financial_metric_count(self, text: str) -> int:
        return len({metric for metric in self.profile.financial_metrics if metric in text})

    def has_structured_source_marker(self, block: DocumentBlock) -> bool:
        source_path = block.source_path.lower()
        return any(marker in source_path for marker in self.profile.structured_source_markers)

    def is_financial_anchor_heading(self, block: DocumentBlock) -> bool:
        text = self.block_text_with_sections(block)
        return self.has_financial_anchor_text(text)

    def contract_bonus(self, block: DocumentBlock) -> float:
        if not self.is_contract_evidence(block):
            return -8.0
        bonus = 8.0
        section_text = " ".join(block.section_path).lower()
        if section_text in {"termination", "term and renewal", "confidentiality"}:
            bonus += 5.0
        if self.is_contract_named_source(block):
            bonus += 5.0
        return bonus

    def is_contract_evidence(self, block: DocumentBlock) -> bool:
        text = self.block_text_with_sections(block)
        contract_sections = {
            "termination",
            "term and renewal",
            "confidentiality",
            "intellectual property",
            "deliverables",
        }
        section_text = " ".join(block.section_path).lower()
        if self.is_contract_named_source(block):
            return True
        if any(section in section_text for section in contract_sections):
            return True
        words = set(text.split())
        return "agreement" in words and bool({"party", "parties"} & words)

    def is_contract_named_source(self, block: DocumentBlock) -> bool:
        return "contract" in block.source_path.lower() or "contract" in block.document_id.lower()

    def is_gross_margin_block(self, block: DocumentBlock) -> bool:
        return "gross margin" in self.block_text_with_sections(block)

    def is_financial_performance_evidence(self, block: DocumentBlock) -> bool:
        if block.block_type.value == "heading":
            return False

        text = self.block_text_with_sections(block)
        section_text = " ".join(block.section_path).lower()
        if self.is_financial_noise_section(section_text):
            return False
        if self.is_financial_risk_prose(text):
            return False
        if self.has_product_service_anchor_text(text):
            return True
        if "gross margin" in section_text and (
            "products" in text or "services" in text or "gross margin" in text
        ):
            return True
        if (
            block.block_type.value == "table"
            and "total net sales" in text
            and "services" in text
            and (
                any(marker in text for marker in self.profile.product_markers)
                or ("products" in text and "gross margin" in text)
            )
        ):
            return True
        if block.block_type.value == "table":
            metric_count = self.financial_metric_count(text)
            if metric_count >= 4 or ("2025" in text and metric_count >= 3):
                return True
        if section_text in self.profile.product_service_sections and "net sales" in text:
            return True
        return False

    def is_product_services_performance_evidence(self, block: DocumentBlock) -> bool:
        if block.block_type.value == "heading":
            return False
        text = self.block_text_with_sections(block)
        section_text = " ".join(block.section_path).lower()
        if self.is_financial_noise_section(section_text) or self.is_financial_risk_prose(text):
            return False
        if self.has_product_service_anchor_text(text):
            return True
        if section_text in self.profile.product_service_sections and "net sales" in text:
            return True
        if "total net sales" not in text:
            return False
        if any(marker in text for marker in self.profile.product_markers):
            return True
        return "products" in text and "services" in text

    def is_financial_noise_section(self, section_text: str) -> bool:
        return any(marker in section_text for marker in self.profile.financial_noise_sections)

    def is_financial_risk_prose(self, text: str) -> bool:
        return any(marker in text for marker in self.profile.financial_risk_markers)

    def is_risk_context(self, block: DocumentBlock) -> bool:
        section_text = " ".join(block.section_path).lower()
        return "risk" in section_text or "risks" in section_text

    def is_generic_heading(self, block: DocumentBlock) -> bool:
        if block.block_type.value != "heading":
            return False
        normalized = block.text.lstrip("#").strip().lower()
        generic_headings = {
            *self.profile.product_service_sections,
            *self.profile.product_markers,
        }
        return normalized in generic_headings or normalized.endswith(" inc.")

    def is_recommendation_query(self, analysis: QueryAnalysis) -> bool:
        recommendation_terms = {
            "recommend",
            "recommends",
            "recommended",
            "recommendation",
        }
        return bool(recommendation_terms & analysis.raw_terms)

    def recommendation_bonus(self, block: DocumentBlock) -> float:
        text = self.block_text_with_sections(block)
        if "recommendation" in text:
            return 18.0
        if "recommended" in text:
            return 10.0
        return 0.0

    def is_low_information_frontmatter(self, block: DocumentBlock) -> bool:
        text = block.text.strip().lower()
        if len(text) > 60:
            return False
        return text in self.profile.low_information_frontmatter

    def has_financial_anchor_text(self, text: str) -> bool:
        return any(anchor in text for anchor in self.profile.financial_anchor_headings)

    def has_product_service_anchor_text(self, text: str) -> bool:
        return any(anchor in text for anchor in self.profile.product_service_anchor_headings)

    def block_text_with_sections(self, block: DocumentBlock) -> str:
        return " ".join([block.text, *block.section_path]).lower()

    def _dedupe(self, candidates: list[DocumentBlock]) -> list[DocumentBlock]:
        unique: list[DocumentBlock] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.block_id in seen:
                continue
            seen.add(candidate.block_id)
            unique.append(candidate)
        return unique
