# MIT License
# Copyright (c) 2024 Cursor AI

import os
import tempfile
from unittest.mock import patch
import pytest
from app import create_app
from app.core.config import Settings, EMBEDDING_DIMENSION
from app.core.vector_store import reset_vector_store
from app.middleware.auth import create_access_token
from langchain_core.embeddings import Embeddings

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts, settings=None):
        return [[float(i % EMBEDDING_DIMENSION) for i in range(EMBEDDING_DIMENSION)] for _ in texts]
    def embed_query(self, text, settings=None):
        return [float(i % EMBEDDING_DIMENSION) for i in range(EMBEDDING_DIMENSION)]

@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_vector_store()
    yield
    reset_vector_store()

@pytest.fixture
def mock_settings():
    return Settings(
        jwt_secret="test-secret-key-for-testing-only",
        jwt_algorithm="HS256", jwt_expiry_hours=24,
        embedding_dimension=EMBEDDING_DIMENSION, top_k=5,
        ollama_base_url="http://localhost:11434",
        ollama_embed_model="nomic-embed-text", ollama_llm_model="tinyllama",
        database_url="sqlite:///data/test.db", log_level="INFO",
        faiss_index_path="data/test_faiss", port=8000, flask_env="testing",
        google_api_key="test-google-key", chunk_size=500, chunk_overlap=50,
        max_context_tokens=4000, llm_model="gpt-4o", score_threshold=0.0,
        embedding_model="models/embedding-001",
    )

@pytest.fixture
def app():
    os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only"
    reset_vector_store()
    fake = FakeEmbeddings()
    with patch("app.core.vector_store.embed_documents", side_effect=fake.embed_documents), \
         patch("app.core.vector_store.embed_query", side_effect=fake.embed_query), \
         patch("app.core.llm.generate_answer", return_value="Refunds take 5 business days."), \
         patch("app.core.llm.generate_answer_stream", return_value=iter(["Refunds take 5 business days."])):
        flask_app = create_app()
        flask_app.config["TESTING"] = True
        yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def auth_headers():
    token = create_access_token()
    return {"Authorization": "Bearer " + token}

@pytest.fixture
def sample_documents():
    from langchain_core.documents import Document
    return [
        Document(page_content="Refunds take 5 to 7 business days.", metadata={"source": "policy.pdf", "page": 1}),
        Document(page_content="Contact support via email for urgent issues.", metadata={"source": "help.pdf", "page": 2}),
    ]

@pytest.fixture
def sample_txt_file():
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, b"Refund Policy: Customers may request a full refund within 14 days. Refunds are processed within 5 business days.")
    os.close(fd)
    yield path
    os.unlink(path)
