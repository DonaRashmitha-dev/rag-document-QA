# MIT License
# Copyright (c) 2026 RAG-QA Contributors

from unittest.mock import MagicMock, patch
import pytest
from app.core.config import EMBEDDING_DIMENSION, Settings
from app.core.embedder import (
    embed_documents, embed_query, ensure_embeddings_object,
    get_embeddings, reset_embeddings_cache,
)

class TestGetEmbeddings:
    def test_returns_gemini_embeddings(self, mock_settings):
        reset_embeddings_cache()
        with patch("app.core.embedder.GoogleGenerativeAIEmbeddings") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            result = get_embeddings(mock_settings)
            assert result is instance
            mock_cls.assert_called_once_with(
                model=mock_settings.embedding_model,
                google_api_key=mock_settings.google_api_key,
                task_type="retrieval_document",
            )

    def test_caches_instance(self, mock_settings):
        reset_embeddings_cache()
        with patch("app.core.embedder.GoogleGenerativeAIEmbeddings") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            first = get_embeddings(mock_settings)
            second = get_embeddings(mock_settings)
            assert first is second
            mock_cls.assert_called_once()

    def test_recreate_on_settings_change(self, mock_settings):
        reset_embeddings_cache()
        with patch("app.core.embedder.GoogleGenerativeAIEmbeddings") as mock_cls:
            mock_cls.side_effect = [MagicMock(), MagicMock()]
            first = get_embeddings(mock_settings)
            new_settings = Settings(
                google_api_key="different-key",
                jwt_secret=mock_settings.jwt_secret,
                embedding_model=mock_settings.embedding_model,
            )
            second = get_embeddings(new_settings)
            assert first is not second
            assert mock_cls.call_count == 2

class TestEmbedDocuments:
    def test_embed_documents_calls_gemini(self, mock_settings):
        fake_vectors = [[0.1] * EMBEDDING_DIMENSION, [0.2] * EMBEDDING_DIMENSION]
        with patch("app.core.embedder.OllamaEmbeddings") as mock_cls:
            instance = MagicMock()
            instance.embed_documents.return_value = fake_vectors
            mock_cls.return_value = instance
            texts = ["hello world", "second document"]
            result = embed_documents(texts, mock_settings)
            assert result == fake_vectors
            instance.embed_documents.assert_called_once_with(texts)

    def test_embed_documents_empty_list(self, mock_settings):
        result = embed_documents([], mock_settings)
        assert result == []

    def test_embed_documents_dimension_validation(self, mock_settings):
        bad_vectors = [[0.1] * 100]
        with patch("app.core.embedder.OllamaEmbeddings") as mock_cls:
            instance = MagicMock()
            instance.embed_documents.return_value = bad_vectors
            mock_cls.return_value = instance
            with pytest.raises(ValueError, match="dimension 100"):
                embed_documents(["short text"], mock_settings)

class TestEmbedQuery:
    def test_embed_query_returns_vector(self, mock_settings):
        fake_vector = [0.5] * EMBEDDING_DIMENSION
        with patch("app.core.embedder.OllamaEmbeddings") as mock_cls:
            instance = MagicMock()
            instance.embed_query.return_value = fake_vector
            mock_cls.return_value = instance
            result = embed_query("test query", mock_settings)
            assert result == fake_vector
            instance.embed_query.assert_called_once_with("test query")

class TestEnsureEmbeddingsObject:
    def test_passes_for_embeddings_instance(self):
        from langchain_core.embeddings import Embeddings
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return []
            def embed_query(self, text):
                return []
        dummy = DummyEmbeddings()
        assert ensure_embeddings_object(dummy) is dummy

    def test_raises_for_non_embeddings(self):
        with pytest.raises(TypeError, match="Expected an Embeddings object"):
            ensure_embeddings_object("not an embeddings object")
