# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Pydantic v2 request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        """Strip leading/trailing whitespace from the query."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class SourceAttribution(BaseModel):
    """Source metadata for a retrieved chunk."""

    doc_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    score: float


class TokenUsage(BaseModel):
    """Token usage breakdown for a query response."""

    prompt: int
    completion: int


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    sources: list[SourceAttribution]
    tokens_used: TokenUsage
    latency_ms: int


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    doc_id: str
    filename: str
    num_chunks: int
    status: str = "indexed"


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    faiss_chunks: int
    uptime_s: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
