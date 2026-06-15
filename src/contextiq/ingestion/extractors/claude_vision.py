"""Claude-vision page extractor.

Renders each PDF page to an image and asks Claude to transcribe the FULL page as
markdown — body text, tables, and descriptions of charts/figures/images (their
labels, numbers, trends). This makes visual evidence retrievable as text, which
the FAST/standard text pipeline cannot do. Pages are processed concurrently.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from contextiq.ingestion.models import BlockType, DocumentBlock

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are extracting one document page for a retrieval index. Output clean "
    "GitHub-flavored markdown of EVERYTHING on the page, in reading order:\n"
    "- body text, verbatim;\n"
    "- every table as a markdown table;\n"
    "- for every chart/figure/diagram/image, a short heading then 1-3 sentences "
    "describing what it shows, including axis labels, series names, key numbers, "
    "and the trend/finding.\n"
    "Output ONLY the markdown for this page. No preamble."
)


def _load_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    for env in (Path(".env"),
                Path("/Volumes/VeN/Claude-Code-Work/projects/contextiq/.env")):
        if env.exists():
            m = re.search(r"ANTHROPIC_API_KEY=(\S+)", env.read_text())
            if m:
                return m.group(1)
    return None


class ClaudeVisionExtractor:
    """Extract pages via Claude vision (markdown incl. figure/chart descriptions)."""

    name = "claude_vision"

    def __init__(self, *, model: str = "claude-sonnet-4-5", scale: float = 2.0,
                 max_workers: int = 6, max_tokens: int = 4000) -> None:
        self.model = model
        self.scale = scale
        self.max_workers = max_workers
        self.max_tokens = max_tokens

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        import pypdfium2 as pdfium  # noqa: PLC0415
        from anthropic import Anthropic  # noqa: PLC0415

        key = _load_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not available for ClaudeVisionExtractor")
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        client = Anthropic(api_key=key)

        pdf = pdfium.PdfDocument(str(path))
        n = len(pdf)
        lo, hi = (1, n) if page_range is None else page_range
        pages = list(range(lo - 1, min(hi, n)))  # 0-indexed

        # render pages -> png bytes (sequential, fast)
        images: dict[int, bytes] = {}
        for i in pages:
            bitmap = pdf[i].render(scale=self.scale)
            pil = bitmap.to_pil()
            import io  # noqa: PLC0415
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images[i] = buf.getvalue()
        pdf.close()

        def transcribe(i: int) -> tuple[int, str]:
            b64 = base64.b64encode(images[i]).decode()
            msg = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": _PROMPT},
                ]}],
            )
            text = "\n".join(b.text for b in msg.content
                             if getattr(b, "type", None) == "text").strip()
            return i, text

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for i, text in ex.map(transcribe, pages):
                results[i] = text

        document_id = self._document_id(path)
        blocks: list[DocumentBlock] = []
        for i in pages:
            page_no = i + 1
            md = results.get(i, "")
            for chunk in (c.strip() for c in md.split("\n\n") if c.strip()):
                idx = len(blocks)
                stripped = chunk.lstrip()
                if stripped.startswith("#"):
                    bt = BlockType.HEADING
                elif "|" in chunk and "---" in chunk:
                    bt = BlockType.TABLE
                else:
                    bt = BlockType.TEXT
                blocks.append(DocumentBlock(
                    document_id=document_id, block_id=f"{document_id}:{idx}",
                    source_path=str(path), page=page_no, block_type=bt, text=chunk,
                    metadata={"parser": "claude_vision", "reading_order": idx,
                              "layout_label": bt.value},
                ))
        return blocks

    def _document_id(self, path: Path) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"{path.stem}-{digest}"
