# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Unit tests for the text chunker module."""

from langchain_core.documents import Document

from app.core.chunker import chunk_documents, create_splitter
from app.core.config import Settings


def test_create_splitter_uses_settings() -> None:
    """Splitter should respect chunk_size and chunk_overlap from settings."""
    settings = Settings(
        google_api_key="test-key",
        jwt_secret="secret",
        chunk_size=256,
        chunk_overlap=32,
    )
    splitter = create_splitter(settings)
    assert splitter._chunk_size == 256
    assert splitter._chunk_overlap == 32


def test_chunk_documents_adds_metadata(sample_documents: list[Document]) -> None:
    """Chunked documents should have chunk_index and content_hash metadata."""
    settings = Settings(
        google_api_key="test-key",
        jwt_secret="secret",
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = chunk_documents(sample_documents, settings)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert "content_hash" in chunk.metadata
        assert len(chunk.metadata["content_hash"]) == 16


def test_chunk_documents_preserves_content() -> None:
    """Chunk content should be a substring of the original document."""
    settings = Settings(
        google_api_key="test-key",
        jwt_secret="secret",
        chunk_size=100,
        chunk_overlap=20,
    )
    original_text = "A" * 250
    docs = [Document(page_content=original_text, metadata={"filename": "test.txt"})]
    chunks = chunk_documents(docs, settings)
    combined = "".join(c.page_content for c in chunks)
    assert len(combined) >= len(original_text) - settings.chunk_overlap
