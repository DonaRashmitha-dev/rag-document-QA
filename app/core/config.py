# MIT License
# Copyright (c) 2024 Cursor AI

from dataclasses import dataclass

EMBEDDING_DIMENSION = 768


@dataclass
class Settings:
    jwt_secret: str = "test-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    embedding_dimension: int = EMBEDDING_DIMENSION
    top_k: int = 10
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "phi3:mini"
    database_url: str = "sqlite:///data/rag.db"
    log_level: str = "INFO"
    faiss_index_path: str = "data/faiss_index"
    port: int = 8000
    flask_env: str = "development"
    google_api_key: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_context_tokens: int = 4000
    llm_model: str = "gpt-4o"
    score_threshold: float = 0.0
    embedding_model: str = "models/embedding-001"


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
