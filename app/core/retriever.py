# MIT License
# Copyright (c) 2024 Cursor AI

from dataclasses import dataclass
from typing import List, Any, Optional
from app.core.config import Settings, get_settings
from app.core.vector_store import get_vector_store


@dataclass
class ScoredDocument:
    text: str = ""
    score: float = 0.0
    source: str = ""
    page: int = 0
    document: Any = None

    def __post_init__(self):
        if self.document is not None:
            if hasattr(self.document, "page_content"):
                self.text = self.document.page_content
            elif isinstance(self.document, dict):
                self.text = self.document.get("text", "")
            else:
                self.text = str(self.document)


def _mmr_rerank(docs, top_k, lambda_mult=0.5):
    if not docs:
        return []
    selected = []
    remaining = list(docs)
    while len(selected) < top_k and remaining:
        best_score = -1.0
        best_idx = 0
        for i, doc in enumerate(remaining):
            relevance = doc.score
            diversity = 0.0
            if selected:
                texts = [d.text for d in selected]
                diversity = min(1.0 - (len(doc.text) / max(len(t), 1)) for t in texts) if texts else 0.0
                diversity = max(0.0, diversity)
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


def deduplicate_chunks(docs, threshold=0.95):
    unique = []
    seen_hashes = set()
    for doc in docs:
        dup = False
        if doc.document is not None:
            meta = getattr(doc.document, "metadata", {})
            if isinstance(meta, dict):
                ch = meta.get("content_hash")
                if ch:
                    if ch in seen_hashes:
                        dup = True
                    else:
                        seen_hashes.add(ch)
        if not dup:
            for u in unique:
                if doc.text == u.text:
                    dup = True
                    break
        if not dup:
            unique.append(doc)
    return unique


def retrieve(query: str, top_k: int = 5, vector_store=None, settings: Optional[Settings] = None):
    cfg = settings or get_settings()
    store = vector_store or get_vector_store(cfg)

    if hasattr(store, "similarity_search_with_scores"):
        raw_results = store.similarity_search_with_scores(query, k=top_k * 2)
        docs = []
        for doc_obj, score in raw_results:
            if hasattr(doc_obj, "page_content"):
                text = doc_obj.page_content
                meta = getattr(doc_obj, "metadata", {})
                source = meta.get("source", meta.get("filename", "")) if isinstance(meta, dict) else ""
                page = meta.get("page", 0) if isinstance(meta, dict) else 0
                docs.append(ScoredDocument(
                    text=text, score=float(score), source=source, page=page, document=doc_obj,
                ))
            elif isinstance(doc_obj, dict):
                text = doc_obj.get("text", "")
                source = doc_obj.get("source", doc_obj.get("filename", ""))
                page = doc_obj.get("page", 0)
                docs.append(ScoredDocument(
                    text=text, score=float(score), source=source, page=page,
                ))
            else:
                docs.append(ScoredDocument(text=str(doc_obj), score=float(score)))
    else:
        raw_results = store.similarity_search(query, k=top_k * 2)
        docs = []
        for r in raw_results:
            docs.append(ScoredDocument(
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                source=r.get("source", ""),
                page=r.get("page", 0),
            ))

    threshold = getattr(cfg, "score_threshold", 0.0)
    if threshold > 0:
        docs = [d for d in docs if d.score >= threshold]

    docs = deduplicate_chunks(docs)
    docs = sorted(docs, key=lambda x: x.score, reverse=True)
    return docs[:top_k]
