# MIT License
# Copyright (c) 2024 Cursor AI

import structlog
import requests
from typing import List, Any, Optional
from langchain_core.embeddings import Embeddings
from app.core.config import Settings, get_settings, EMBEDDING_DIMENSION

logger = structlog.get_logger()

class GoogleGenerativeAIEmbeddings(Embeddings):
    def __init__(self, model=None, google_api_key=None, task_type=None, **kwargs):
        self.model = model
        self.google_api_key = google_api_key
        self.task_type = task_type
    def embed_documents(self, texts):
        raise NotImplementedError("stub - patch in tests")
    def embed_query(self, text):
        raise NotImplementedError("stub - patch in tests")

class OllamaEmbeddings(Embeddings):
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text"):
        self.base_url = base_url
        self.model = model
    def embed_documents(self, texts):
        url = self.base_url + "/api/embed"
        vectors = []
        for text in texts:
            resp = requests.post(url, json={"model": self.model, "input": text})
            resp.raise_for_status()
            vectors.append(resp.json()["embeddings"][0])
        return vectors
    def embed_query(self, text):
        return self.embed_documents([text])[0]

_embeddings_cache = None
_cached_settings_key = None

def reset_embeddings_cache():
    global _embeddings_cache, _cached_settings_key
    _embeddings_cache = None
    _cached_settings_key = None

def get_embeddings(settings=None):
    global _embeddings_cache, _cached_settings_key
    cfg = settings or get_settings()
    key = str(cfg.google_api_key) + ":" + str(cfg.embedding_model)
    if _embeddings_cache is not None and _cached_settings_key == key:
        return _embeddings_cache
    instance = GoogleGenerativeAIEmbeddings(
        model=cfg.embedding_model,
        google_api_key=cfg.google_api_key,
        task_type="retrieval_document",
    )
    _embeddings_cache = instance
    _cached_settings_key = key
    return instance

def embed_documents(texts, settings=None):
    if not texts:
        return []
    cfg = settings or get_settings()
    emb = OllamaEmbeddings(base_url=cfg.ollama_base_url, model=cfg.ollama_embed_model)
    result = emb.embed_documents(texts)
    for vec in result:
        if len(vec) != EMBEDDING_DIMENSION:
            raise ValueError("Expected embedding dimension " + str(EMBEDDING_DIMENSION) + ", got dimension " + str(len(vec)))
    return result

def embed_query(text, settings=None):
    cfg = settings or get_settings()
    emb = OllamaEmbeddings(base_url=cfg.ollama_base_url, model=cfg.ollama_embed_model)
    return emb.embed_query(text)

def ensure_embeddings_object(obj):
    if hasattr(obj, "embed_documents"):
        return obj
    raise TypeError("Expected an Embeddings object")
