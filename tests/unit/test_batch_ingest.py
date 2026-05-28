from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.batch import BatchIngestor, renumber_blocks
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.ingestion.profiles import FAST
from contextiq.jobs.models import IngestJobCreate, IngestJobStatus
from contextiq.jobs.runner import run_ingest_job
from contextiq.jobs.store import IngestJobStore
from contextiq.retrieval.store import LocalDocumentStore


def test_renumber_blocks_assigns_sequential_ids() -> None:
    blocks = [
        DocumentBlock(
            document_id="old",
            block_id="old:0",
            source_path="doc.pdf",
            text="one",
        ),
        DocumentBlock(
            document_id="old",
            block_id="old:1",
            source_path="doc.pdf",
            text="two",
        ),
    ]

    renumbered, next_index = renumber_blocks(
        blocks,
        document_id="doc-abc",
        start_index=5,
    )

    assert next_index == 7
    assert [block.block_id for block in renumbered] == ["doc-abc:5", "doc-abc:6"]


def test_batch_ingestor_uses_single_pass_for_markdown(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Heading\n\nBody", encoding="utf-8")

    result = BatchIngestor(profile=FAST).ingest(path)

    assert result.batches_run == 1
    assert result.profile_name == "fast"
    assert result.blocks
    assert result.blocks[0].block_type == BlockType.HEADING


def test_batch_ingestor_batches_large_pdf_by_page_range(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.pdf"
    path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "contextiq.ingestion.batch.count_pdf_pages",
        lambda _path: 120,
    )

    calls: list[tuple[int, int]] = []

    def fake_load_pdf_range(self, pdf_path: Path, *, page_range: tuple[int, int]):
        calls.append(page_range)
        return [
            DocumentBlock(
                document_id="ignored",
                block_id=f"ignored:{page_range[0]}",
                source_path=str(pdf_path),
                page=page_range[0],
                text=f"batch {page_range[0]}-{page_range[1]}",
            )
        ]

    monkeypatch.setattr(
        "contextiq.ingestion.loader.DocumentLoader.load_pdf_range",
        fake_load_pdf_range,
    )

    result = BatchIngestor(profile=FAST).ingest(path)

    assert result.batches_run == 3
    assert result.pages_total == 120
    assert result.pages_done == 120
    assert calls == [(1, 50), (51, 100), (101, 120)]
    assert len(result.blocks) == 3
    assert result.blocks[0].block_id.endswith(":0")


def test_run_ingest_job_persists_blocks_and_status(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sample.md"
    source.write_text("# Title\n\nEvidence text.", encoding="utf-8")
    store_path = tmp_path / "blocks.json"
    jobs_db = tmp_path / "jobs.db"

    job_store = IngestJobStore(db_path=jobs_db)
    job = job_store.create(
        IngestJobCreate(source_path=str(source), profile_name="fast", build_index=False)
    )

    completed = run_ingest_job(
        job.job_id,
        job_store=job_store,
        document_store=LocalDocumentStore(path=store_path),
    )

    assert completed.status == IngestJobStatus.COMPLETED
    assert completed.blocks_saved >= 2
    assert completed.document_id is not None
    assert LocalDocumentStore(path=store_path).stats()["blocks"] >= 2


def test_store_load_blocks_can_scope_to_one_document(tmp_path: Path) -> None:
    store_path = tmp_path / "blocks.json"
    store = LocalDocumentStore(path=store_path)
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc-a",
                block_id="doc-a:1",
                source_path="a.md",
                text="alpha",
            ),
            DocumentBlock(
                document_id="doc-b",
                block_id="doc-b:1",
                source_path="b.md",
                text="beta",
            ),
        ]
    )

    scoped = store.load_blocks(document_id="doc-a")

    assert len(scoped) == 1
    assert scoped[0].document_id == "doc-a"
