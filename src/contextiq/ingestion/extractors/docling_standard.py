"""Docling-backed extractor — implements the Extractor protocol."""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
from pathlib import Path
from typing import Any

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.profiles import QUALITY, IngestProfile

logger = logging.getLogger(__name__)


class DoclingStandardExtractor:
    """Extract document blocks from PDFs using Docling."""

    name = "docling_standard"

    def __init__(
        self,
        *,
        profile: IngestProfile | None = None,
        visuals_dir: Path | None = None,
        enable_picture_enrichment: bool | None = None,
    ) -> None:
        self.profile = profile or QUALITY
        self.visuals_dir = visuals_dir or Path("data/processed/visuals")
        if enable_picture_enrichment is None:
            self.enable_picture_enrichment = self.profile.enable_picture_enrichment
        else:
            self.enable_picture_enrichment = enable_picture_enrichment

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        """Return ordered blocks for the document (optionally a page range)."""
        return self._load_with_docling(path, page_range=page_range)

    def _load_with_docling(
        self,
        path: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> list[DocumentBlock]:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_options = PdfPipelineOptions()
        pdf_options.generate_page_images = self.profile.generate_page_images
        pdf_options.generate_picture_images = self.profile.generate_picture_images
        pdf_options.do_picture_classification = self.enable_picture_enrichment
        pdf_options.do_picture_description = self.enable_picture_enrichment
        if self.profile.table_mode_fast:
            pdf_options.table_structure_options.mode = TableFormerMode.FAST

        if self.enable_picture_enrichment:
            self._enable_docling_picture_enrichment(pdf_options)

        converter = self._docling_converter(
            input_format=InputFormat,
            pdf_format_option=PdfFormatOption,
            pdf_options=pdf_options,
            document_converter=DocumentConverter,
        )
        convert_kwargs: dict[str, tuple[int, int]] = {}
        if page_range is not None:
            convert_kwargs["page_range"] = page_range
        try:
            result = converter.convert(str(path), **convert_kwargs)
        except Exception:
            if not self.enable_picture_enrichment:
                raise
            logger.warning("Docling picture enrichment failed; retrying basic conversion")
            fallback_options = PdfPipelineOptions()
            fallback_options.generate_page_images = self.profile.generate_page_images
            fallback_options.generate_picture_images = self.profile.generate_picture_images
            if self.profile.table_mode_fast:
                fallback_options.table_structure_options.mode = TableFormerMode.FAST
            converter = self._docling_converter(
                input_format=InputFormat,
                pdf_format_option=PdfFormatOption,
                pdf_options=fallback_options,
                document_converter=DocumentConverter,
            )
            result = converter.convert(str(path), **convert_kwargs)
        return self._load_docling_document(document=result.document, path=path)

    def _enable_docling_picture_enrichment(self, pdf_options: Any) -> None:
        for name, value in {
            "do_picture_classification": True,
            "do_picture_description": True,
            "images_scale": 2,
        }.items():
            if hasattr(pdf_options, name):
                setattr(pdf_options, name, value)

    def _docling_converter(
        self,
        *,
        input_format: Any,
        pdf_format_option: Any,
        pdf_options: Any,
        document_converter: Any,
    ) -> Any:
        return document_converter(
            format_options={
                input_format.PDF: pdf_format_option(pipeline_options=pdf_options),
            }
        )

    def _load_docling_document(self, document: Any, path: Path) -> list[DocumentBlock]:
        document_id = self._document_id(path)
        blocks: list[DocumentBlock] = []
        section_stack: list[str] = []

        for item, _level in document.iterate_items():
            label = self._label_value(getattr(item, "label", "text"))
            block_type = self._block_type_from_label(label)
            text = self._text_from_item(item=item, block_type=block_type, document=document)
            if not text:
                continue

            block_index = len(blocks)
            metadata = {
                "parser": "docling",
                "docling_ref": str(getattr(item, "self_ref", "")),
                "docling_label": label,
                **self._visual_metadata(
                    item=item,
                    block_type=block_type,
                    document=document,
                    document_id=document_id,
                    block_index=block_index,
                ),
            }
            metadata["reading_order"] = block_index
            metadata["layout_label"] = label
            if block_type == BlockType.HEADING:
                metadata["heading_level"] = max(int(getattr(item, "level", 1) or 1), 1)

            if block_type == BlockType.HEADING:
                heading = text.lstrip("#").strip()
                heading_level = max(int(getattr(item, "level", 1) or 1), 1)
                section_stack = section_stack[: heading_level - 1]
                section_stack.append(heading)
                block_text = f"{'#' * min(heading_level + 1, 6)} {heading}"
            else:
                block_text = self._text_with_visual_description(
                    text=text,
                    metadata=metadata,
                )

            blocks.append(
                DocumentBlock(
                    document_id=document_id,
                    block_id=f"{document_id}:{block_index}",
                    source_path=str(path),
                    page=self._page_from_item(item),
                    section_path=section_stack.copy(),
                    block_type=block_type,
                    text=block_text,
                    metadata=metadata,
                )
            )

        return blocks

    def _text_from_item(self, item: Any, block_type: BlockType, document: Any | None = None) -> str:
        if block_type == BlockType.TABLE and hasattr(item, "export_to_markdown"):
            return str(item.export_to_markdown(doc=document)).strip()
        if block_type == BlockType.FIGURE:
            caption = self._caption_text(item=item, document=document)
            return f"Figure: {caption}".strip() if caption else "<figure>"
        return str(getattr(item, "text", "")).strip()

    def _visual_metadata(
        self,
        item: Any,
        block_type: BlockType,
        document: Any | None = None,
        document_id: str | None = None,
        block_index: int | None = None,
    ) -> dict[str, str | int | float | bool | None]:
        if block_type == BlockType.FIGURE:
            caption = self._caption_text(item=item, document=document)
            image_path = self._save_visual_image(
                item=item,
                document=document,
                document_id=document_id,
                block_index=block_index,
            )
            return {
                "visual_kind": "figure",
                "caption": str(caption).strip() if caption else None,
                "image_path": image_path,
                **self._docling_picture_metadata(item),
                **self._bbox_metadata(item),
            }
        if block_type == BlockType.TABLE:
            image_path = self._save_visual_image(
                item=item,
                document=document,
                document_id=document_id,
                block_index=block_index,
            )
            return {
                "visual_kind": "table",
                "image_path": image_path,
                **self._bbox_metadata(item),
            }
        return {}

    def _docling_picture_metadata(
        self, item: Any
    ) -> dict[str, str | int | float | bool | None]:
        description, provider = self._docling_picture_description(item)
        visual_class, confidence, classes = self._docling_picture_classes(item)
        return {
            "visual_description": description,
            "visual_description_provider": provider,
            "visual_class": visual_class,
            "visual_class_confidence": confidence,
            "visual_classes": classes,
        }

    def _text_with_visual_description(
        self,
        *,
        text: str,
        metadata: dict[str, str | int | float | bool | None],
    ) -> str:
        if metadata.get("visual_kind") != "figure":
            return text
        visual_description = metadata.get("visual_description")
        if not visual_description:
            return text
        return f"{text}\nVisual description: {visual_description}"

    def _docling_picture_description(self, item: Any) -> tuple[str | None, str | None]:
        meta_description = getattr(getattr(item, "meta", None), "description", None)
        if meta_description is not None:
            text = self._clean_visual_text(getattr(meta_description, "text", ""))
            provider = str(getattr(meta_description, "created_by", "")).strip()
            return text or None, provider or None

        for annotation in self._docling_annotations(item):
            if self._annotation_kind(annotation) != "description":
                continue
            text = self._clean_visual_text(getattr(annotation, "text", ""))
            provider = str(getattr(annotation, "provenance", "")).strip()
            return text or None, provider or None
        return None, None

    def _clean_visual_text(self, text: Any) -> str:
        cleaned = str(text or "").split("<end_of", maxsplit=1)[0]
        return " ".join(cleaned.split())

    def _docling_picture_classes(
        self, item: Any
    ) -> tuple[str | None, float | None, str | None]:
        classification = getattr(getattr(item, "meta", None), "classification", None)
        if classification is not None:
            predictions = list(getattr(classification, "predictions", []) or [])
            main_prediction = self._main_prediction(classification, predictions)
            return self._classification_metadata(main_prediction, predictions)

        for annotation in self._docling_annotations(item):
            if self._annotation_kind(annotation) != "classification":
                continue
            predictions = list(getattr(annotation, "predicted_classes", []) or [])
            main_prediction = predictions[0] if predictions else None
            return self._classification_metadata(main_prediction, predictions)
        return None, None, None

    def _main_prediction(self, classification: Any, predictions: list[Any]) -> Any | None:
        get_main_prediction = getattr(classification, "get_main_prediction", None)
        if callable(get_main_prediction):
            return get_main_prediction()
        return predictions[0] if predictions else None

    def _classification_metadata(
        self,
        main_prediction: Any | None,
        predictions: list[Any],
    ) -> tuple[str | None, float | None, str | None]:
        visual_class = self._prediction_class_name(main_prediction)
        confidence = self._prediction_confidence(main_prediction)
        classes = ", ".join(
            self._format_prediction(prediction)
            for prediction in predictions
            if self._prediction_class_name(prediction)
        )
        return visual_class, confidence, classes or None

    def _format_prediction(self, prediction: Any) -> str:
        class_name = self._prediction_class_name(prediction)
        confidence = self._prediction_confidence(prediction)
        if confidence is None:
            return str(class_name)
        return f"{class_name}:{confidence:.2f}"

    def _prediction_class_name(self, prediction: Any | None) -> str | None:
        if prediction is None:
            return None
        class_name = getattr(prediction, "class_name", None)
        return str(class_name).strip() if class_name is not None else None

    def _prediction_confidence(self, prediction: Any | None) -> float | None:
        if prediction is None:
            return None
        confidence = getattr(prediction, "confidence", None)
        return float(confidence) if confidence is not None else None

    def _docling_annotations(self, item: Any) -> list[Any]:
        get_annotations = getattr(item, "get_annotations", None)
        if callable(get_annotations):
            return list(get_annotations())
        return list(getattr(item, "annotations", []) or [])

    def _annotation_kind(self, annotation: Any) -> str:
        return str(getattr(annotation, "kind", "")).lower()

    def _page_from_item(self, item: Any) -> int | None:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            return None
        # Try first provenance item
        page_no = getattr(provenance[0], "page_no", None)
        if page_no is not None:
            return int(page_no)
        # Fallback: try other provenance items
        for prov in provenance[1:]:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                return int(page_no)
        # Fallback: try bbox-based page inference if available
        bbox = getattr(provenance[0], "bbox", None)
        if bbox is not None and hasattr(bbox, "page_no"):
            return int(bbox.page_no)
        return None

    def _caption_text(self, item: Any, document: Any | None = None) -> str | None:
        caption = getattr(item, "caption_text", None)
        if callable(caption):
            caption = self._call_caption_text(caption=caption, document=document)
        if caption is None:
            return None
        text = str(caption).strip()
        return text or None

    def _call_caption_text(self, caption: Any, document: Any | None) -> Any:
        try:
            signature = inspect.signature(caption)
        except (TypeError, ValueError):
            return caption(document) if document is not None else caption()

        required_params = [
            param
            for param in signature.parameters.values()
            if param.default is inspect.Parameter.empty
            and param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if required_params and document is not None:
            return caption(document)
        return caption()

    def _save_visual_image(
        self,
        item: Any,
        document: Any | None,
        document_id: str | None,
        block_index: int | None,
    ) -> str | None:
        if document is None or document_id is None or block_index is None:
            return None
        get_image = getattr(item, "get_image", None)
        if not callable(get_image):
            return None
        try:
            image = get_image(document)
        except Exception as exc:
            logger.warning("Docling visual image extraction failed", exc_info=exc)
            return None
        if image is None:
            return None

        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", document_id).strip("-") or "document"
        output_dir = self.visuals_dir / slug[:80]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"block-{block_index:05d}.png"
        try:
            image.save(output_path)
        except Exception as exc:
            logger.warning("Docling visual image save failed", exc_info=exc)
            return None
        return str(output_path)

    def _bbox_metadata(self, item: Any) -> dict[str, str | int | float | bool | None]:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            return {}
        bbox = getattr(provenance[0], "bbox", None)
        if bbox is None:
            return {}
        return {
            "bbox_l": self._float_attr(bbox, "l"),
            "bbox_t": self._float_attr(bbox, "t"),
            "bbox_r": self._float_attr(bbox, "r"),
            "bbox_b": self._float_attr(bbox, "b"),
            "bbox_coord_origin": str(getattr(bbox, "coord_origin", "")) or None,
        }

    def _float_attr(self, item: Any, name: str) -> float | None:
        value = getattr(item, name, None)
        return float(value) if value is not None else None

    def _block_type_from_label(self, label: str) -> BlockType:
        if label in {"section_header", "title"}:
            return BlockType.HEADING
        if label == "table":
            return BlockType.TABLE
        if label in {"picture", "figure"}:
            return BlockType.FIGURE
        return BlockType.TEXT

    def _label_value(self, label: Any) -> str:
        return str(getattr(label, "value", label)).lower()

    def _document_id(self, path: Path) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"{path.stem}-{digest}"
