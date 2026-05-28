"""FastAPI application for ContextIQ."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from contextiq.context.engine import ContextEngine
from contextiq.context.models import ContextPacket
from contextiq.core.config import get_settings
from contextiq.ingestion.batch import BatchIngestor
from contextiq.ingestion.profiles import FAST, QUALITY
from contextiq.jobs.models import IngestJob, IngestJobCreate, IngestJobStatus
from contextiq.jobs.runner import run_ingest_job
from contextiq.jobs.store import IngestJobStore
from contextiq.llm.answerer import GroundedAnswerer
from contextiq.retrieval.store import LocalDocumentStore

LIGHTWEIGHT_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ContextIQ</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0c0f;
      --panel: #17191f;
      --panel-2: #20232b;
      --border: #343844;
      --text: #f4f5f7;
      --muted: #a7adbb;
      --accent: #f05a14;
      --good: #54d18a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      background: #101116;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1, h2, h3 { margin: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 16px; margin-bottom: 10px; }
    h3 { font-size: 15px; overflow-wrap: anywhere; }
    .muted { color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: minmax(380px, 460px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      max-width: 1900px;
      margin: 0 auto;
    }
    aside, section, .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    aside { padding: 14px; height: fit-content; position: sticky; top: 78px; }
    section { padding: 14px; }
    label { display: block; color: var(--muted); margin: 12px 0 6px; font-size: 12px; }
    input, textarea, select {
      width: 100%;
      color: var(--text);
      background: #0f1116;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      padding: 11px 12px;
      margin-top: 12px;
      cursor: pointer;
    }
    button.secondary { background: #555863; }
    details {
      border-top: 1px solid var(--border);
      margin-top: 14px;
      padding-top: 12px;
    }
    summary { cursor: pointer; color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(110px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .kpi {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }
    .kpi span { display: block; color: var(--muted); font-size: 12px; }
    .kpi strong { font-size: 20px; }
    .answer {
      margin-bottom: 14px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .source-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 12px;
    }
    .source {
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      min-width: 0;
    }
    .badges { margin: 8px 0; }
    .badge {
      display: inline-block;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #252934;
      color: var(--text);
      padding: 2px 8px;
      margin: 0 4px 5px 0;
      font-size: 12px;
    }
    .badge.accent { color: #ffb07b; border-color: #8a4b2a; }
    .visual-thumb {
      display: block;
      width: 100%;
      max-height: 300px;
      object-fit: contain;
      background: #0d0f13;
      border: 1px solid var(--border);
      border-radius: 6px;
      margin: 10px 0;
    }
    pre {
      overflow-x: auto;
      white-space: pre;
      background: #0d0f13;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      max-height: 320px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .status { margin-top: 10px; color: var(--muted); min-height: 20px; }
    @media (max-width: 1000px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; }
      .source-grid { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ContextIQ</h1>
      <div class="muted">Complex document retrieval without UI theatrics.</div>
    </div>
    <div id="stats" class="muted">Loading stats...</div>
  </header>
  <main>
    <aside>
      <h2>Ask</h2>
      <label for="question">Question</label>
      <textarea id="question" placeholder="Ask a question about the indexed corpus"></textarea>
      <div class="controls">
        <div>
          <label for="limit">Sources</label>
          <input id="limit" type="number" min="1" max="40" value="6" />
        </div>
        <div>
          <label for="budget">Token budget</label>
          <input id="budget" type="number" min="500" max="50000" value="6000" />
        </div>
      </div>
      <button onclick="answerQuestion()">Answer with Evidence</button>
      <button class="secondary" onclick="buildContext()">Build Context Only</button>
      <button class="secondary" onclick="refreshStats()">Refresh Stats</button>
      <details>
        <summary>Ingest document</summary>
        <label for="document">PDF, Markdown, text, or Excel</label>
        <input id="document" type="file" />
        <label>
          <input id="index" type="checkbox" checked style="width:auto" /> Build vector index
        </label>
        <button onclick="ingestDocument()">Ingest</button>
      </details>
      <div id="status" class="status"></div>
    </aside>
    <div>
      <section>
        <div id="answer" class="answer muted">Ask a question to generate a grounded answer.</div>
        <div id="sources" class="source-grid"></div>
      </section>
    </div>
  </main>
  <script>
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);

    function setStatus(message) {
      document.getElementById("status").textContent = message;
    }

    function sourcePreview(source) {
      const lines = String(source.text || "").split("\n").filter((line) => {
        const stripped = line.replaceAll("|", "").trim();
        return line.trim() && !/^-+$/.test(stripped.replaceAll(" ", ""));
      });
      return lines.slice(0, 8).join("\n");
    }

    function renderPacket(data) {
      document.getElementById("answer").innerHTML = `
        <div class="kpis">
          <div class="kpi"><span>Sources</span><strong>${data.sources.length}</strong></div>
          <div class="kpi"><span>Used tokens</span><strong>${data.used_tokens}</strong></div>
          <div class="kpi"><span>Budget</span><strong>${data.token_budget}</strong></div>
          <div class="kpi"><span>Dropped</span><strong>${data.dropped_candidates}</strong></div>
        </div>
        <h2>${esc(data.question)}</h2>
        <pre>${esc(data.sources[0] ? sourcePreview(data.sources[0]) : "No sources.")}</pre>
      `;
      renderSources(data.sources);
    }

    function renderAnswer(data) {
      const context = data.context;
      const warnings = (data.warnings || []).map((warning) =>
        `<span class="badge">${esc(warning)}</span>`
      ).join("");
      document.getElementById("answer").innerHTML = `
        <div class="kpis">
          <div class="kpi"><span>Sources</span><strong>${context.sources.length}</strong></div>
          <div class="kpi"><span>Used tokens</span><strong>${context.used_tokens}</strong></div>
          <div class="kpi"><span>Answer mode</span><strong>${esc(data.mode)}</strong></div>
          <div class="kpi"><span>Model</span><strong>${esc(data.model)}</strong></div>
        </div>
        <h2>${esc(data.question)}</h2>
        <pre>${esc(data.answer)}</pre>
        <div class="badges">${warnings}</div>
      `;
      renderSources(context.sources);
    }

    function renderSources(sources) {
      document.getElementById("sources").innerHTML = sources.map((source) => {
        const stages = (source.stages || []).join(", ") || "unknown";
        const section = (source.section_path || []).join(" > ") || "section unknown";
        const score = Number.isFinite(source.score) ? source.score.toFixed(2) : "unknown";
        const visual = source.metadata?.visual_kind
          ? `<span class="badge">${esc(source.metadata.visual_kind)}</span>`
          : "";
        const visualClass = source.metadata?.visual_class
          ? `<span class="badge">${esc(source.metadata.visual_class)}</span>`
          : "";
        const visualDescription = source.metadata?.visual_description
          ? `<p><strong>Docling visual description:</strong> ${
              esc(source.metadata.visual_description)
            }</p>`
          : "";
        const visualImage = source.metadata?.image_path
          ? `<img class="visual-thumb" src="/visuals?path=${
              encodeURIComponent(source.metadata.image_path)
            }" alt="Visual evidence for ${esc(source.block_id)}" loading="lazy" />`
          : "";
        return `
          <article class="source">
            <h3>${esc(source.block_id)}</h3>
            <div class="badges">
              <span class="badge accent">${esc(source.block_type)}</span>
              <span class="badge">score ${esc(score)}</span>
              <span class="badge">${esc(stages)}</span>
              <span class="badge">page ${esc(source.page || "unknown")}</span>
              ${visual}
              ${visualClass}
            </div>
            <p class="muted">${esc(section)}</p>
            <p>${esc(source.reason)}</p>
            ${visualImage}
            ${visualDescription}
            <pre>${esc(source.text)}</pre>
          </article>
        `;
      }).join("");
    }

    async function buildContext() {
      setStatus("Retrieving...");
      const response = await fetch("/context", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          question: document.getElementById("question").value,
          token_budget: Number(document.getElementById("budget").value),
          limit: Number(document.getElementById("limit").value)
        })
      });
      if (!response.ok) {
        setStatus(`Request failed: ${response.status}`);
        return;
      }
      renderPacket(await response.json());
      setStatus("Done.");
    }

    async function answerQuestion() {
      setStatus("Retrieving and synthesizing...");
      const response = await fetch("/answer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          question: document.getElementById("question").value,
          token_budget: Number(document.getElementById("budget").value),
          limit: Number(document.getElementById("limit").value)
        })
      });
      if (!response.ok) {
        setStatus(`Request failed: ${response.status}`);
        return;
      }
      renderAnswer(await response.json());
      setStatus("Done.");
    }

    async function refreshStats() {
      const response = await fetch("/stats");
      const stats = await response.json();
      document.getElementById("stats").textContent =
        `${stats.documents} documents · ${stats.blocks} blocks`;
    }

    async function ingestDocument() {
      const file = document.getElementById("document").files[0];
      if (!file) {
        setStatus("Choose a file first.");
        return;
      }
      setStatus("Ingesting...");
      const form = new FormData();
      form.append("file", file);
      const index = document.getElementById("index").checked;
      const response = await fetch(`/ingest?index=${index}`, {method: "POST", body: form});
      if (!response.ok) {
        setStatus(`Ingest failed: ${response.status}`);
        return;
      }
      const data = await response.json();
      setStatus(`Ingested ${data.blocks} blocks, indexed ${data.indexed}.`);
      refreshStats();
    }

    refreshStats();
  </script>
</body>
</html>
"""


class HealthResponse(BaseModel):
    """Backend health response."""

    status: str = "ok"
    service: str = "contextiq"


class StoreStatsResponse(BaseModel):
    """Document store statistics."""

    documents: int
    blocks: int


class IngestResponse(BaseModel):
    """Document ingestion response."""

    filename: str
    stored_path: str
    blocks: int
    indexed: int
    profile_name: str
    pages_total: int
    batches_run: int


class IngestJobResponse(BaseModel):
    """Background ingest job response."""

    job_id: str
    status: IngestJobStatus
    source_path: str


class IngestJobStatusResponse(BaseModel):
    """Detailed ingest job status."""

    job_id: str
    status: IngestJobStatus
    source_path: str
    profile_name: str
    pages_total: int
    pages_done: int
    blocks_saved: int
    blocks_indexed: int
    document_id: str | None
    error: str | None


class ContextRequest(BaseModel):
    """Context packet request."""

    question: str = Field(min_length=1)
    token_budget: int = Field(default=6_000, ge=500, le=50_000)
    limit: int = Field(default=24, ge=1, le=100)


class ContextSourceResponse(BaseModel):
    """Selected context source."""

    block_id: str
    document_id: str
    page: int | None
    section_path: list[str]
    block_type: str
    estimated_tokens: int
    reason: str
    stages: list[str]
    score: float | None
    metadata: dict[str, str | int | float | bool | None]
    text: str


class ContextResponse(BaseModel):
    """Context packet response."""

    question: str
    markdown: str
    token_budget: int
    used_tokens: int
    dropped_candidates: int
    sources: list[ContextSourceResponse]


class AnswerResponse(BaseModel):
    """Grounded answer response."""

    question: str
    answer: str
    model: str
    mode: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    warnings: list[str]
    context: ContextResponse


def _context_response(packet: ContextPacket) -> ContextResponse:
    """Serialize a context packet for the API."""

    return ContextResponse(
        question=packet.question,
        markdown=packet.to_markdown(),
        token_budget=packet.token_budget,
        used_tokens=packet.used_tokens,
        dropped_candidates=packet.dropped_candidates,
        sources=[
            ContextSourceResponse(
                block_id=source.block.block_id,
                document_id=source.block.document_id,
                page=source.block.page,
                section_path=source.block.section_path,
                block_type=source.block.block_type.value,
                estimated_tokens=source.estimated_tokens,
                reason=source.reason,
                stages=source.stages,
                score=source.score,
                metadata=source.block.metadata,
                text=source.block.text,
            )
            for source in packet.sources
        ],
    )


def _resolve_visual_artifact(path: str) -> Path:
    """Resolve a processed visual artifact path without allowing traversal."""

    requested = Path(path)
    allowed_root = Path("data/processed/visuals").resolve()
    resolved = requested.resolve()
    if allowed_root != resolved and allowed_root not in resolved.parents:
        raise HTTPException(status_code=404, detail="Visual artifact not found.")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Visual artifact not found.")
    return resolved


def _ingest_job_response(job: IngestJob) -> IngestJobStatusResponse:
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        source_path=job.source_path,
        profile_name=job.profile_name,
        pages_total=job.pages_total,
        pages_done=job.pages_done,
        blocks_saved=job.blocks_saved,
        blocks_indexed=job.blocks_indexed,
        document_id=job.document_id,
        error=job.error,
    )


def create_app(
    store_path: Path | None = None,
    answerer_factory: Callable[[], GroundedAnswerer] | None = None,
) -> FastAPI:
    """Create the ContextIQ FastAPI app."""

    app = FastAPI(
        title="ContextIQ API",
        version="0.1.0",
        description="Context-engineered document intelligence backend.",
    )

    def store() -> LocalDocumentStore:
        return LocalDocumentStore(path=store_path)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(LIGHTWEIGHT_DASHBOARD_HTML)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StoreStatsResponse)
    def stats() -> StoreStatsResponse:
        return StoreStatsResponse(**store().stats())

    @app.get("/visuals")
    def visual_artifact(path: str) -> FileResponse:
        return FileResponse(_resolve_visual_artifact(path))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_document(
        file: Annotated[UploadFile, File()],
        index: bool = True,
        profile: str | None = None,
    ) -> IngestResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        settings = get_settings()
        raw_dir = settings.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        destination = raw_dir / Path(file.filename).name

        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        selected_profile = None
        if profile == "fast":
            selected_profile = FAST
        elif profile == "quality":
            selected_profile = QUALITY

        result = BatchIngestor(profile=selected_profile).ingest(destination)
        document_store = store()
        document_store.save_blocks(result.blocks)
        indexed = document_store.index_blocks(result.blocks) if index else 0

        return IngestResponse(
            filename=file.filename,
            stored_path=str(destination),
            blocks=len(result.blocks),
            indexed=indexed,
            profile_name=result.profile_name,
            pages_total=result.pages_total,
            batches_run=result.batches_run,
        )

    @app.post("/ingest/async", response_model=IngestJobResponse)
    async def ingest_document_async(
        file: Annotated[UploadFile, File()],
        background_tasks: BackgroundTasks,
        index: bool = True,
        profile: str | None = None,
    ) -> IngestJobResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        settings = get_settings()
        raw_dir = settings.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        destination = raw_dir / Path(file.filename).name

        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        job_store = IngestJobStore()
        job = job_store.create(
            IngestJobCreate(
                source_path=str(destination.resolve()),
                profile_name=profile,
                build_index=index,
            )
        )
        background_tasks.add_task(run_ingest_job, job.job_id)
        return IngestJobResponse(
            job_id=job.job_id,
            status=job.status,
            source_path=job.source_path,
        )

    @app.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
    def ingest_job_status(job_id: str) -> IngestJobStatusResponse:
        job = IngestJobStore().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found.")
        return _ingest_job_response(job)

    @app.post("/context", response_model=ContextResponse)
    def build_context(request: ContextRequest) -> ContextResponse:
        packet = ContextEngine(store=store(), token_budget=request.token_budget).build_context(
            request.question,
            limit=request.limit,
        )
        return _context_response(packet)

    @app.post("/answer", response_model=AnswerResponse)
    def answer_question(request: ContextRequest) -> AnswerResponse:
        packet = ContextEngine(store=store(), token_budget=request.token_budget).build_context(
            request.question,
            limit=request.limit,
        )
        answerer = answerer_factory() if answerer_factory else GroundedAnswerer()
        answer = answerer.answer(packet)
        return AnswerResponse(
            question=packet.question,
            answer=answer.text,
            model=answer.model,
            mode=answer.mode,
            tokens_in=answer.tokens_in,
            tokens_out=answer.tokens_out,
            cost_usd=answer.cost_usd,
            warnings=answer.warnings,
            context=_context_response(packet),
        )

    return app


app = create_app()


def run() -> None:
    """Run the API server."""

    uvicorn.run("contextiq.api.main:app", host="127.0.0.1", port=8000, reload=True)
