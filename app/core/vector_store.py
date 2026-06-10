# MIT License
# Copyright (c) 2024 Cursor AI

import json
import os
import uuid

import faiss
import numpy as np
import structlog

from app.core.config import get_settings
from app.core.embedder import embed_documents, embed_query

logger = structlog.get_logger()


class VectorStoreManager:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.index_path = os.path.join("data", "faiss_index")
        self.metadata_path = os.path.join("data", "faiss_metadata.json")
        self.dimension = self.settings.embedding_dimension
        self.index = None
        self.metadata = []
        self._load()

    def chunk_count(self):
        return self.index.ntotal if self.index else 0

    def _load(self):
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info("faiss_index_loaded", count=len(self.metadata))
            except Exception as exc:
                logger.warning("faiss_load_failed", error=str(exc))
                self._create_empty()
        else:
            self._create_empty()

    def _create_empty(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("creating_empty_faiss_index", dimension=self.dimension)

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info("faiss_index_persisted")

    def add_documents(self, chunks, doc_id=None):
        if not chunks:
            return 0
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        texts = []
        for c in chunks:
            if hasattr(c, "page_content"):
                texts.append(c.page_content)
            elif isinstance(c, dict):
                texts.append(c.get("text", ""))
            else:
                texts.append(str(c))
        vectors = embed_documents(texts, self.settings)
        vectors_np = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)
        self.index.add(vectors_np)
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, "page_content"):
                text = chunk.page_content
                meta = getattr(chunk, "metadata", {})
                source = meta.get("source", "") if isinstance(meta, dict) else ""
                page = meta.get("page", 0) if isinstance(meta, dict) else 0
            elif isinstance(chunk, dict):
                text = chunk.get("text", "")
                source = chunk.get("source", "")
                page = chunk.get("page", 0)
            else:
                text = str(chunk)
                source = ""
                page = 0
            self.metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": text,
                "source": source,
                "page": page,
            })
        self._save()
        return len(chunks)

    def similarity_search(self, query, k=5):
        query_vec = embed_query(query, self.settings)
        query_np = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_np)
        scores, indices = self.index.search(query_np, k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def similarity_search_with_scores(self, query, k=5):
        results = self.similarity_search(query, k=k)
        return [(r, r.pop("score", 0.0)) for r in results]

    def get_stats(self):
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
        }


_vector_store_instance = None


def get_vector_store(settings=None):
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager(settings)
    return _vector_store_instance


def reset_vector_store():
    global _vector_store_instance
    _vector_store_instance = None
