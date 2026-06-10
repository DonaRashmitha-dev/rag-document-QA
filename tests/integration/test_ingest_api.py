# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Integration tests for POST /ingest."""

from io import BytesIO


def test_ingest_requires_auth(client) -> None:
    """POST /ingest without JWT should return 401."""
    data = {"file": (BytesIO(b"test content"), "test.txt")}
    response = client.post("/ingest", data=data, content_type="multipart/form-data")
    assert response.status_code == 401


def test_ingest_txt_file(client, auth_headers: dict[str, str], sample_txt_file: str) -> None:
    """POST /ingest with a valid text file should index and return doc metadata."""
    with open(sample_txt_file, "rb") as file_handle:
        data = {"file": (file_handle, "policy.txt")}
        response = client.post(
            "/ingest",
            data=data,
            headers=auth_headers,
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "indexed"
    assert body["filename"] == "policy.txt"
    assert body["num_chunks"] > 0
    assert len(body["doc_id"]) == 8


def test_ingest_unsupported_type(client, auth_headers: dict[str, str]) -> None:
    """POST /ingest with unsupported file type should return 400."""
    data = {"file": (BytesIO(b"binary"), "malware.exe")}
    response = client.post(
        "/ingest",
        data=data,
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["error"]


def test_ingest_no_file(client, auth_headers: dict[str, str]) -> None:
    """POST /ingest without file field should return 400."""
    response = client.post("/ingest", headers=auth_headers)
    assert response.status_code == 400
