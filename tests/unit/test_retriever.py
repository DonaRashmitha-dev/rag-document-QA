# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Unit tests for the retriever module."""

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.retriever import ScoredDocument, _mmr_rerank, deduplicate_chunks, retrieve


def test_deduplicate_chunks_removes_duplicates() -> None:
    """Deduplication should keep only the first occurrence of each content hash."""
    doc1 = ScoredDocument(
        document=Document(page_content="same text", metadata={"content_hash": "abc123"}),
        score=0.9,
    )
    doc2 = ScoredDocument(
        document=Document(page_content="same text", metadata={"content_hash": "abc123"}),
        score=0.8,
    )
    doc3 = ScoredDocument(
        document=Document(page_content="different text", metadata={"content_hash": "def456"}),
        score=0.7,
    )
    result = deduplicate_chunks([doc1, doc2, doc3])
    assert len(result) == 2
    assert result[0].score == 0.9


def test_mmr_rerank_limits_results() -> None:
    """MMR reranking should return at most top_k documents."""
    candidates = [
        ScoredDocument(
            document=Document(page_content=f"chunk {i} unique content number {i}"),
            score=0.9 - i * 0.05,
        )
        for i in range(10)
    ]
    result = _mmr_rerank(candidates, top_k=3)
    assert len(result) == 3


def test_retrieve_with_mock_vector_store() -> None:
    """retrieve should filter by score threshold and return scored documents."""
    settings = Settings(
        google_api_key="test-key",
        jwt_secret="secret",
        score_threshold=0.5,
        top_k=2,
    )

    mock_doc = Document(
        page_content="Refund within 14 days.",
        metadata={"filename": "policy.txt", "chunk_index": 0, "content_hash": "aaa"},
    )

    from unittest.mock import MagicMock

    mock_store = MagicMock()
    mock_store.similarity_search_with_scores.return_value = [
        (mock_doc, 0.95),
        (Document(page_content="low score", metadata={"content_hash": "bbb"}), 0.3),
    ]

    results = retrieve("refund policy", settings=settings, vector_store=mock_store, top_k=2)
    assert len(results) == 1
    assert results[0].score == 0.95
