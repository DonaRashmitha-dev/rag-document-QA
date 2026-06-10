# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Integration tests for POST /query."""

import json
from io import BytesIO


def _ingest_sample(client, auth_headers: dict[str, str]) -> None:
    """Helper to ingest a sample document before querying."""
    content = (
        b"Refund Policy: Customers may request a full refund within 14 days. "
        b"Refunds are processed within 5 business days."
    )
    data = {"file": (BytesIO(content), "policy.txt")}
    client.post(
        "/ingest",
        data=data,
        headers=auth_headers,
        content_type="multipart/form-data",
    )


def test_query_requires_auth(client) -> None:
    """POST /query without JWT should return 401."""
    response = client.post(
        "/query",
        json={"query": "What is the refund policy?"},
    )
    assert response.status_code == 401


def test_query_valid_request(client, auth_headers: dict[str, str]) -> None:
    """POST /query should return answer, sources, and token usage."""
    _ingest_sample(client, auth_headers)

    response = client.post(
        "/query",
        json={"query": "What is the refund policy?", "top_k": 3},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.get_json()
    assert "answer" in body
    assert isinstance(body["sources"], list)
    assert "tokens_used" in body
    assert "latency_ms" in body
    assert body["tokens_used"]["prompt"] >= 0


def test_query_validation_error(client, auth_headers: dict[str, str]) -> None:
    """POST /query with empty query should return 400."""
    response = client.post(
        "/query",
        json={"query": ""},
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_health_endpoint(client) -> None:
    """GET /health should return status without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] in ("ok", "degraded")
    assert "faiss_chunks" in body
    assert "uptime_s" in body


def test_query_sse_streaming(client, auth_headers: dict[str, str]) -> None:
    """POST /query with SSE accept header should stream tokens."""
    _ingest_sample(client, auth_headers)

    response = client.post(
        "/query",
        json={"query": "What is the refund policy?"},
        headers={**auth_headers, "Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.content_type.startswith("text/event-stream")
    data_lines = [
        line for line in response.data.decode().split("\n") if line.startswith("data: ")
    ]
    assert len(data_lines) >= 1
    final_event = json.loads(data_lines[-1].replace("data: ", ""))
    assert final_event.get("done") is True
