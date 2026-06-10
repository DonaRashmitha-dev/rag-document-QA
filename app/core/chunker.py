# MIT License
# Copyright (c) 2024 Cursor AI

import hashlib

from langchain_core.documents import Document

from app.core.config import Settings, get_settings


class _Splitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap


def create_splitter(settings: Settings | None = None):
    cfg = settings or get_settings()
    return _Splitter(cfg.chunk_size, cfg.chunk_overlap)


def chunk_documents(documents, settings: Settings | None = None) -> list[Document]:
    cfg = settings or get_settings()
    if isinstance(documents, str):
        documents = [Document(page_content=documents, metadata={})]
    elif not isinstance(documents, list):
        documents = [documents]

    splitter = create_splitter(cfg)
    chunk_size = splitter._chunk_size
    chunk_overlap = splitter._chunk_overlap
    all_chunks: list[Document] = []

    for doc in documents:
        text = doc.page_content
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end]
            content_hash = hashlib.md5(piece.encode()).hexdigest()[:16]
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            meta["content_hash"] = content_hash
            all_chunks.append(Document(page_content=piece, metadata=meta))
            start += chunk_size - chunk_overlap
            idx += 1
            if end == len(text):
                break

    return all_chunks
