# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Unit tests for prompt building and token budget management."""

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.prompt import (
    apply_token_budget,
    build_messages,
    count_tokens,
    format_chunk_for_context,
)
from app.core.retriever import ScoredDocument


def test_count_tokens_positive() -> None:
    """Token count should be positive for non-empty text."""
    assert count_tokens("Hello, world!") > 0


def test_format_chunk_for_context() -> None:
    """Formatted chunk should include filename, page, and score."""
    scored = ScoredDocument(
        document=Document(
            page_content="Refund within 14 days.",
            metadata={"filename": "policy.pdf", "page": 3, "chunk_index": 12},
        ),
        score=0.91,
    )
    formatted = format_chunk_for_context(scored)
    assert "policy.pdf" in formatted
    assert "14 days" in formatted
    assert "0.91" in formatted


def test_apply_token_budget_drops_low_scoring() -> None:
    """Token budget should drop lowest-scoring chunks when over budget."""
    settings = Settings(
        google_api_key="test-key",
        jwt_secret="secret",
        max_context_tokens=100,
    )
    docs = [
        ScoredDocument(
            document=Document(
                page_content="A" * 500,
                metadata={"filename": "a.txt", "chunk_index": i},
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    result = apply_token_budget(docs, "short query", settings)
    assert len(result) < len(docs)


def test_build_messages_structure() -> None:
    """build_messages should return SystemMessage and HumanMessage."""
    settings = Settings(google_api_key="test-key", jwt_secret="secret")
    scored = ScoredDocument(
        document=Document(
            page_content="Refunds within 14 days.",
            metadata={"filename": "policy.txt", "chunk_index": 0},
        ),
        score=0.9,
    )
    messages = build_messages("What is the refund policy?", [scored], settings)
    assert len(messages) == 2
    assert "refund policy" in messages[1].content.lower()  # type: ignore[union-attr]
    assert "14 days" in messages[1].content  # type: ignore[union-attr]
