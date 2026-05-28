from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from contextiq.api.main import create_app
from contextiq.jobs.store import IngestJobStore


def test_api_async_ingest_queues_job(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "blocks.json"
    jobs_db = tmp_path / "jobs.db"

    def job_store_factory() -> IngestJobStore:
        return IngestJobStore(db_path=jobs_db)

    monkeypatch.setattr("contextiq.api.main.IngestJobStore", job_store_factory)
    monkeypatch.setattr("contextiq.api.main.run_ingest_job", lambda job_id: None)

    client = TestClient(create_app(store_path=store_path))
    response = client.post(
        "/ingest/async?index=false",
        files={"file": ("sample.md", BytesIO(b"# Title\n\nBody"), "text/markdown")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["job_id"]

    status = client.get(f"/ingest/{payload['job_id']}")
    assert status.status_code == 200
    assert status.json()["source_path"].endswith("sample.md")
