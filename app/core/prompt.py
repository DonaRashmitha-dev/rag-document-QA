# MIT License
# Copyright (c) 2024 Cursor AI


from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.retriever import ScoredDocument

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer using ONLY the context provided below. "
    "Do NOT use any prior knowledge or make up information. "
    "If the answer is not in the context, say exactly: "
    "'I could not find this information in the provided documents.' "
    "Keep your answer concise: 2-3 sentences maximum. "
    "Cite the source filename for every claim."
)

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def format_chunk_for_context(doc: ScoredDocument) -> str:
    meta = {}
    if doc.document is not None:
        meta = getattr(doc.document, "metadata", {}) or {}
    filename = meta.get("filename", meta.get("source", "unknown"))
    page = meta.get("page", "?")
    score = round(doc.score, 2)
    return "[" + str(filename) + " p." + str(page) + " score=" + str(score) + "]\n" + doc.text

def apply_token_budget(docs: list[ScoredDocument], query: str, settings=None) -> list[ScoredDocument]:
    cfg = settings or get_settings()
    budget = cfg.max_context_tokens
    available = budget - count_tokens(query) - count_tokens(SYSTEM_PROMPT) - 200
    sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
    selected = []
    used = 0
    for doc in sorted_docs:
        chunk_tokens = count_tokens(format_chunk_for_context(doc))
        if used + chunk_tokens > available:
            break
        selected.append(doc)
        used += chunk_tokens
    return selected

def build_messages(query: str, docs: list[ScoredDocument], settings=None) -> list:
    context = "\n\n".join(format_chunk_for_context(d) for d in docs)
    human_text = "Context:\n" + context + "\n\nQuestion: " + query
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_text)]

