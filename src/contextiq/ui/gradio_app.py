"""Gradio frontend for ContextIQ."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)(?:,\s*page\s+(?:\d+|unknown))?\]")

APP_CSS = """
.gradio-container { max-width: none !important; }
.contextiq-shell { max-width: 1500px; margin: 0 auto; }
.contextiq-title h1 { margin-bottom: 0.15rem; }
.contextiq-title p { color: #a7a7ad; margin-top: 0; }
.compact-panel {
  border: 1px solid #34363b;
  border-radius: 8px;
  background: #18191d;
  padding: 10px 12px;
}
.query-row textarea { min-height: 74px !important; }
.answer-card, .source-card {
  border: 1px solid #3a3a40;
  border-radius: 8px;
  background: #17181b;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.answer-kpis {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 8px;
  margin: 12px 0;
}
.kpi {
  border: 1px solid #34363b;
  border-radius: 6px;
  padding: 8px 10px;
  background: #202126;
}
.kpi span { display: block; color: #a7a7ad; font-size: 12px; }
.kpi strong { font-size: 18px; }
.badge {
  display: inline-block;
  border: 1px solid #454852;
  border-radius: 999px;
  padding: 2px 8px;
  margin: 0 4px 4px 0;
  font-size: 12px;
  color: #f4f4f5;
  background: #24262d;
}
.badge.hot { border-color: #ff7a1a; color: #ffb47a; }
.muted { color: #a7a7ad; }
.source-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}
.source-card { margin-bottom: 0; }
.source-card h3 {
  overflow-wrap: anywhere;
  margin-top: 0;
  font-size: 17px;
}
.source-card p {
  line-height: 1.55;
}
.visual-thumb {
  display: block;
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  border: 1px solid #33353a;
  border-radius: 8px;
  background: #0f1012;
  margin: 10px 0 12px;
}
.source-text {
  overflow-x: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border: 1px solid #33353a;
  border-radius: 6px;
  padding: 14px;
  background: #0f1012;
  max-height: 520px;
  font-size: 14px;
  line-height: 1.6;
}
.table-wrap {
  overflow: auto;
  border: 1px solid #33353a;
  border-radius: 8px;
  background: #0f1012;
  max-height: 620px;
}
.source-table {
  width: 100%;
  min-width: 900px;
  border-collapse: collapse;
  font-size: 14px;
  line-height: 1.45;
}
.source-table th,
.source-table td {
  border: 1px solid #383a40;
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
  white-space: normal;
  overflow-wrap: anywhere;
}
.source-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #202126;
  color: #f4f4f5;
}
.source-table td {
  color: #e6e6e9;
}
.answer-body {
  overflow-x: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border: 1px solid #33353a;
  border-radius: 6px;
  padding: 14px;
  background: #0f1012;
  max-height: 640px;
  font-size: 14px;
  line-height: 1.65;
  margin: 0;
}
.grounding-list {
  list-style: none;
  padding: 0;
  margin: 8px 0 0;
  display: grid;
  gap: 6px;
}
.grounding-list li {
  border: 1px solid #34363b;
  border-radius: 6px;
  padding: 8px 10px;
  background: #202126;
  font-size: 13px;
  line-height: 1.45;
}
.grounding-list .page-num {
  color: #ffb47a;
  font-weight: 600;
}
.source-card.cited {
  border-color: #ff7a1a;
}
@media (max-width: 900px) {
  .answer-kpis { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .source-list { grid-template-columns: 1fr; }
  .source-table { min-width: 720px; }
}
"""


def _backend_url(value: str) -> str:
    return value.rstrip("/") if value else DEFAULT_BACKEND_URL


def ingest_file(file_path: str | None, backend_url: str, build_index: bool) -> str:
    """Upload a document through the FastAPI backend."""

    if not file_path:
        return "Upload a document first."

    path = Path(file_path)
    with path.open("rb") as file:
        response = httpx.post(
            f"{_backend_url(backend_url)}/ingest",
            params={"index": build_index},
            files={"file": (path.name, file, "application/octet-stream")},
            timeout=300,
        )
    response.raise_for_status()
    data = response.json()
    return (
        f"Ingested {data['blocks']} blocks from `{data['filename']}`.\n\n"
        f"Indexed {data['indexed']} blocks.\n\n"
        f"Stored at `{data['stored_path']}`."
    )


def ask_question(
    question: str,
    backend_url: str,
    token_budget: int,
    limit: int,
    block_types: list[str] | None,
) -> tuple[str, str, str]:
    """Ask the backend for a grounded answer plus retrieval trace."""

    if not question.strip():
        return _empty_answer_panel(), "Ask a question first.", ""

    response = httpx.post(
        f"{_backend_url(backend_url)}/answer",
        json={"question": question, "token_budget": int(token_budget), "limit": int(limit)},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    context = data.get("context") or {}
    sources = context.get("sources") or []
    cited_ids = _cited_block_ids(str(data.get("answer", "")))
    visible_sources = _visible_sources(sources, block_types)
    return (
        _format_answer_panel(data, visible_sources, cited_ids=cited_ids),
        str(context.get("markdown", "")),
        _format_source_cards(
            sources,
            block_types,
            backend_url=backend_url,
            cited_ids=cited_ids,
        ),
    )


def _cited_block_ids(answer: str) -> set[str]:
    return {match.group(1) for match in _CITATION_RE.finditer(answer)}


def _source_sort_key(source: dict[str, Any], cited_ids: set[str]) -> tuple[bool, bool, str]:
    block_id = str(source.get("block_id", ""))
    return (block_id not in cited_ids, source.get("page") is None, block_id)


def _visible_sources(
    sources: list[dict[str, Any]],
    block_types: list[str] | None,
) -> list[dict[str, Any]]:
    allowed = set(block_types or [])
    return [
        source
        for source in sources
        if not allowed or str(source.get("block_type", "")) in allowed
    ]


def _empty_answer_panel() -> str:
    return (
        '<div class="answer-card">'
        "<h2>ContextIQ Answer</h2>"
        '<p class="muted">Ask a question first.</p>'
        "</div>"
    )


def _format_answer_panel(
    data: dict[str, Any],
    visible_sources: list[dict[str, Any]],
    *,
    cited_ids: set[str],
) -> str:
    context = data.get("context") or {}
    question = escape(str(data.get("question", "")))
    answer_text = str(data.get("answer", "")).strip() or "No answer returned."
    mode = escape(str(data.get("mode", "unknown")))
    model = escape(str(data.get("model", "unknown")))
    tokens_in = escape(str(data.get("tokens_in", "unknown")))
    tokens_out = escape(str(data.get("tokens_out", "unknown")))
    used_tokens = escape(str(context.get("used_tokens", "unknown")))
    warnings = [
        escape(str(item))
        for item in data.get("warnings") or []
        if str(item).strip()
    ]
    warnings_html = (
        "<p class=\"muted\">" + "<br />".join(warnings) + "</p>" if warnings else ""
    )

    return f"""
<div class="answer-card">
  <h2>ContextIQ Answer</h2>
  <p class="muted">{question}</p>
  <div class="answer-kpis">
    <div class="kpi"><span>Sources</span><strong>{len(visible_sources)}</strong></div>
    <div class="kpi"><span>Mode</span><strong>{mode}</strong></div>
    <div class="kpi"><span>Tokens in / out</span><strong>{tokens_in} / {tokens_out}</strong></div>
    <div class="kpi"><span>Context tokens</span><strong>{used_tokens}</strong></div>
  </div>
  <p><span class="badge">{model}</span></p>
  <pre class="answer-body">{escape(answer_text)}</pre>
  {_format_grounding_section(visible_sources, cited_ids=cited_ids)}
  {warnings_html}
</div>
""".strip()


def _format_grounding_section(
    sources: list[dict[str, Any]],
    *,
    cited_ids: set[str],
) -> str:
    grounding = [s for s in sources if str(s.get("block_id", "")) in cited_ids] or sources
    if not grounding:
        return '<p class="muted">No retrieval grounding available.</p>'

    items: list[str] = []
    for source in sorted(grounding, key=lambda s: _source_sort_key(s, cited_ids)):
        block_id = str(source.get("block_id", "unknown"))
        page = source.get("page")
        page_label = str(page) if page is not None else "unknown"
        section = " > ".join(source.get("section_path") or []) or "section unknown"
        cited = ' <span class="badge hot">cited</span>' if block_id in cited_ids else ""
        items.append(
            "<li>"
            f'<span class="page-num">page {escape(page_label)}</span>'
            f" · {escape(block_id)}{cited}"
            f'<br /><span class="muted">{escape(section)}</span>'
            "</li>"
        )

    return (
        "<h3>Grounding (page trace)</h3>"
        f'<ul class="grounding-list">{"".join(items)}</ul>'
    )


def _format_source_cards(
    sources: list[dict[str, Any]],
    block_types: list[str] | None,
    backend_url: str = DEFAULT_BACKEND_URL,
    *,
    cited_ids: set[str] | None = None,
) -> str:
    visible_sources = _visible_sources(sources, block_types)
    if not visible_sources:
        return '<div class="source-card muted">No sources match the selected filters.</div>'

    cited = cited_ids or set()
    cards: list[str] = ['<div class="source-list">']
    for source in sorted(visible_sources, key=lambda s: _source_sort_key(s, cited)):
        metadata = source.get("metadata") or {}
        section = " > ".join(source.get("section_path") or []) or "section unknown"
        stages = ", ".join(source.get("stages") or []) or "unknown"
        page = source.get("page")
        page_label = str(page) if page is not None else "unknown"
        score = source.get("score")
        score_text = f"{float(score):.2f}" if isinstance(score, int | float) else "unknown"
        block_id = str(source.get("block_id", "unknown"))
        is_cited = block_id in cited
        caption = metadata.get("caption")
        visual_kind = metadata.get("visual_kind")
        visual_description = metadata.get("visual_description")
        visual_class = metadata.get("visual_class")
        image_path = metadata.get("image_path")
        visual = ""
        if visual_kind or caption or visual_class:
            visual = (
                f'<span class="badge">{escape(str(visual_kind or "visual"))}'
                f'{f" | {escape(str(visual_class))}" if visual_class else ""}'
                f'{f" | {escape(str(caption))}" if caption else ""}</span>'
            )
        visual_description_html = ""
        if visual_description:
            visual_description_html = (
                "<p><strong>Docling visual description:</strong> "
                f"{escape(str(visual_description))}</p>"
            )
        visual_image_html = ""
        if image_path:
            visual_url = _visual_artifact_url(backend_url=backend_url, image_path=str(image_path))
            visual_image_html = (
                f'<img class="visual-thumb" src="{escape(visual_url)}" '
                f'alt="Visual evidence for {escape(block_id)}" '
                'loading="lazy" />'
            )
        evidence = _format_source_evidence(
            str(source.get("text", "")),
            str(source.get("block_type", "")),
        )
        cited_badge = '<span class="badge hot">cited in answer</span>' if is_cited else ""
        card_class = "source-card cited" if is_cited else "source-card"
        cards.append(
            f"""
<div class="{card_class}">
  <h3>{escape(block_id)}</h3>
  <p>
    <span class="badge hot">{escape(str(source.get("block_type", "unknown")))}</span>
    <span class="badge">page {escape(page_label)}</span>
    <span class="badge">score {escape(score_text)}</span>
    <span class="badge">{escape(stages)}</span>
    {cited_badge}
    {visual}
  </p>
  <p class="muted">{escape(section)}</p>
  <p>{escape(str(source.get("reason", "unknown")))}</p>
  {visual_image_html}
  {visual_description_html}
  {evidence}
</div>
""".strip()
        )

    cards.append("</div>")
    return "\n".join(cards)


def _visual_artifact_url(*, backend_url: str, image_path: str) -> str:
    return f"{_backend_url(backend_url)}/visuals?path={quote(image_path, safe='/')}"


def _format_source_evidence(text: str, block_type: str) -> str:
    rows = _markdown_table_rows(text)
    if block_type == "table" and rows:
        return _format_table(rows)
    if _looks_like_markdown_table(text) and rows:
        return _format_table(rows)
    return f'<pre class="source-text">{escape(text)}</pre>'


def _looks_like_markdown_table(text: str) -> bool:
    pipe_rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    return len(pipe_rows) >= 2


def _markdown_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])

    if not rows:
        return []

    column_count = max(len(row) for row in rows)
    return [row + [""] * (column_count - len(row)) for row in rows]


def _format_table(rows: list[list[str]]) -> str:
    header, *body = rows
    header_html = "".join(f"<th>{escape(cell)}</th>" for cell in header)
    body_html = "\n".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return f"""
<div class="table-wrap">
  <table class="source-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{body_html}</tbody>
  </table>
</div>
""".strip()


def load_stats(backend_url: str) -> str:
    """Load backend store stats."""

    response = httpx.get(f"{_backend_url(backend_url)}/stats", timeout=20)
    response.raise_for_status()
    data = response.json()
    return f"Documents: {data['documents']} | Blocks: {data['blocks']}"


def build_app() -> Any:
    """Build the Gradio Blocks app."""

    import gradio as gr

    with gr.Blocks(title="ContextIQ", elem_classes=["contextiq-shell"]) as demo:
        gr.Markdown(
            "# ContextIQ\n"
            "Document intelligence for filings, architecture spreadsheets, contracts, "
            "tables, and visual evidence.",
            elem_classes=["contextiq-title"],
        )

        with gr.Row():
            backend_url = gr.Textbox(
                value=DEFAULT_BACKEND_URL,
                label="Backend URL",
                scale=3,
            )
            stats_button = gr.Button("Refresh Stats", scale=1)
        stats_output = gr.Markdown()

        with gr.Accordion("Ingest documents", open=False):
            with gr.Row():
                document = gr.File(
                    label="Document",
                    file_types=[".pdf", ".md", ".txt", ".xlsx", ".xlsm", ".xltx", ".xltm"],
                    scale=3,
                )
                with gr.Column(scale=1, min_width=260):
                    build_index = gr.Checkbox(value=True, label="Build vector index")
                    ingest_button = gr.Button("Ingest", variant="primary")
            ingest_output = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=4):
                question = gr.Textbox(
                    value="",
                    label="Question",
                    placeholder="Ask a question about the indexed corpus",
                    lines=2,
                    elem_classes=["query-row"],
                )
            with gr.Column(scale=1, min_width=260):
                ask_button = gr.Button(
                    "Ask",
                    variant="primary",
                )

        with gr.Accordion("Retrieval controls", open=False):
            with gr.Row():
                token_budget = gr.Slider(
                    minimum=1_000,
                    maximum=20_000,
                    value=6_000,
                    step=500,
                    label="Token budget",
                )
                limit = gr.Slider(
                    minimum=1,
                    maximum=40,
                    value=6,
                    step=1,
                    label="Source limit",
                )
            block_types = gr.CheckboxGroup(
                choices=["text", "table", "figure", "heading"],
                value=["text", "table", "figure", "heading"],
                label="Source types",
            )

        answer_output = gr.HTML(label="Answer")
        with gr.Tabs():
            with gr.Tab("Sources"):
                source_output = gr.HTML(label="Source Trace")
            with gr.Tab("Context Packet"):
                context_output = gr.Code(
                    label="Context Packet",
                    language="markdown",
                    lines=24,
                )

        ingest_button.click(
            fn=ingest_file,
            inputs=[document, backend_url, build_index],
            outputs=ingest_output,
        )
        stats_button.click(fn=load_stats, inputs=backend_url, outputs=stats_output)
        ask_button.click(
            fn=ask_question,
            inputs=[question, backend_url, token_budget, limit, block_types],
            outputs=[answer_output, context_output, source_output],
        )

    return demo


def main() -> None:
    """Run the Gradio UI."""

    from contextiq.obs import init_obs

    init_obs()  # wires ven_obs -> Langfuse + ven_master.db (rollup-only if no LF keys)

    build_app().launch(server_name="127.0.0.1", server_port=7860, css=APP_CSS)
