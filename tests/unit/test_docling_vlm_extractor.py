from __future__ import annotations

from contextiq.ingestion.extractors.docling_vlm import DoclingVLMExtractor
from contextiq.ingestion.extractors.stub import StubExtractor
from contextiq.ingestion.models import BlockType, DocumentBlock


def test_vlm_selects_mlx_on_apple_silicon() -> None:
    ext = DoclingVLMExtractor(has_mps=True, mlx_available=True)
    assert ext.vlm_model_name == "GRANITEDOCLING_MLX"


def test_vlm_selects_transformers_without_mps() -> None:
    ext = DoclingVLMExtractor(has_mps=False)
    assert ext.vlm_model_name == "GRANITEDOCLING_TRANSFORMERS"


def test_vlm_falls_back_to_standard_on_conversion_error(tmp_path) -> None:
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    fallback = StubExtractor([DocumentBlock(
        document_id="d", block_id="d:0", source_path=str(pdf),
        block_type=BlockType.TEXT, text="standard fallback block",
    )])
    ext = DoclingVLMExtractor(has_mps=True, fallback=fallback)

    def boom(_path, *, page_range=None):
        raise RuntimeError("vlm model unavailable")

    ext._convert = boom  # type: ignore[method-assign]

    blocks = ext.extract(pdf)
    assert blocks[0].text == "standard fallback block"
    assert ext.name == "docling_vlm"


def test_vlm_falls_back_to_transformers_when_mlx_runtime_missing() -> None:
    # The bug we hit live: MPS present but mlx_vlm not installed must NOT select
    # MLX (which silently degrades ~10x), it must pick the Transformers spec.
    ext = DoclingVLMExtractor(has_mps=True, mlx_available=False)
    assert ext.vlm_model_name == "GRANITEDOCLING_TRANSFORMERS"
