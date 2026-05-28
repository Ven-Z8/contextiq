"""SQLite-backed ingest job store."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from contextiq.core.config import get_settings
from contextiq.jobs.models import IngestJob, IngestJobCreate, IngestJobStatus


class IngestJobStore:
    """Persist and query ingest job status."""

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self.db_path = db_path or settings.jobs_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    build_index INTEGER NOT NULL,
                    pages_total INTEGER NOT NULL DEFAULT 0,
                    pages_done INTEGER NOT NULL DEFAULT 0,
                    blocks_saved INTEGER NOT NULL DEFAULT 0,
                    blocks_indexed INTEGER NOT NULL DEFAULT 0,
                    document_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, request: IngestJobCreate) -> IngestJob:
        job = IngestJob(
            job_id=str(uuid.uuid4()),
            source_path=request.source_path,
            profile_name=request.profile_name or "fast",
            build_index=request.build_index,
        )
        self.save(job)
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def save(self, job: IngestJob) -> None:
        job.updated_at = datetime.now(tz=UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs (
                    job_id, source_path, status, profile_name, build_index,
                    pages_total, pages_done, blocks_saved, blocks_indexed,
                    document_id, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    status = excluded.status,
                    profile_name = excluded.profile_name,
                    build_index = excluded.build_index,
                    pages_total = excluded.pages_total,
                    pages_done = excluded.pages_done,
                    blocks_saved = excluded.blocks_saved,
                    blocks_indexed = excluded.blocks_indexed,
                    document_id = excluded.document_id,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    job.job_id,
                    job.source_path,
                    job.status.value,
                    job.profile_name,
                    int(job.build_index),
                    job.pages_total,
                    job.pages_done,
                    job.blocks_saved,
                    job.blocks_indexed,
                    job.document_id,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )

    def _row_to_job(self, row: sqlite3.Row) -> IngestJob:
        return IngestJob(
            job_id=row["job_id"],
            source_path=row["source_path"],
            status=IngestJobStatus(row["status"]),
            profile_name=row["profile_name"],
            build_index=bool(row["build_index"]),
            pages_total=row["pages_total"],
            pages_done=row["pages_done"],
            blocks_saved=row["blocks_saved"],
            blocks_indexed=row["blocks_indexed"],
            document_id=row["document_id"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
