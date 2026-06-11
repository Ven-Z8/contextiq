"""Docling VLM pipeline extractor (granite-docling-258M, MLX on Apple Silicon)."""

from __future__ import annotations

import logging
from pathlib import Path

from contextiq.ingestion.extractors.base import Extractor
from contextiq.ingestion.extractors.docling_standard import DoclingStandardExtractor
from contextiq.ingestion.models import DocumentBlock

logger = logging.getLogger(__name__)


def _detect_mps() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class DoclingVLMExtractor:
    """VLM-based extraction with automatic fallback to the standard pipeline."""

    name = "docling_vlm"

    def __init__(
        self,
        *,
        has_mps: bool | None = None,
        fallback: Extractor | None = None,
    ) -> None:
        self._has_mps = _detect_mps() if has_mps is None else has_mps
        self.vlm_model_name = (
            "GRANITEDOCLING_MLX" if self._has_mps else "GRANITEDOCLING_TRANSFORMERS"
        )
        self._fallback = fallback or DoclingStandardExtractor()
        self._standard_walk = DoclingStandardExtractor()

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        try:
            return self._convert(path, page_range=page_range)
        except Exception as exc:
            logger.warning("VLM extraction failed (%s); falling back to standard", exc)
            return self._fallback.extract(path, page_range=page_range)

    def _convert(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        from docling.datamodel import vlm_model_specs  # noqa: PLC0415
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import VlmPipelineOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
        from docling.pipeline.vlm_pipeline import VlmPipeline  # noqa: PLC0415

        options = VlmPipelineOptions(
            vlm_options=getattr(vlm_model_specs, self.vlm_model_name)
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=options
                )
            }
        )
        convert_kwargs = {} if page_range is None else {"page_range": page_range}
        result = converter.convert(str(path), **convert_kwargs)
        return self._standard_walk._load_docling_document(
            document=result.document, path=path
        )
