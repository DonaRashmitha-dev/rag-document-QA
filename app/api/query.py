# MIT License
# Copyright (c) 2024 Cursor AI

import json
import time
import structlog
from flask import Blueprint, request, Response, jsonify
from app.core.config import get_settings
from app.core.retriever import retrieve
from app.core.llm import generate_answer, generate_answer_stream
from app.middleware.auth import token_required
from app.middleware.rate_limit import rate_limit

logger = structlog.get_logger()
query_bp = Blueprint("query", __name__, url_prefix="/query")


def _is_quota_error(error_text):
    error_lower = error_text.lower()
    keywords = ["quota", "rate limit", "429", "exceeded", "too many requests", "resource exhausted"]
    for k in keywords:
        if k in error_lower:
            return True
    return False


def _estimate_tokens(text):
    return len(text.split())


@query_bp.route("", methods=["POST"])
@token_required
@rate_limit
def query_document():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        stream = data.get("stream", False) or request.headers.get("Accept", "").startswith("text/event-stream")
        cfg = get_settings()
        docs = retrieve(query, top_k=cfg.top_k)

        if not docs:
            latency = round((time.time() - start_time) * 1000, 2)
            resp_body = {
                "answer": "No relevant documents found.",
                "sources": [],
                "tokens_used": {"prompt": 0, "completion": 0},
                "latency_ms": latency,
                "done": True,
            }
            if stream:
                def empty_stream():
                    payload = json.dumps(resp_body)
                    yield "data: " + payload + "\n\n"
                return Response(empty_stream(), mimetype="text/event-stream", content_type="text/event-stream")
            return jsonify(resp_body), 200

        context_parts = []
        for i, doc in enumerate(docs):
            idx = str(i + 1)
            context_parts.append("[Source " + idx + "] " + doc.text)
        context = "\n\n".join(context_parts)

        system_msg = "You are a helpful assistant. Use the provided context to answer the question. Cite sources like [Source 1], [Source 2]."
        user_msg = "Context:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer:"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        if stream:
            def event_stream():
                try:
                    for chunk in generate_answer_stream(messages):
                        payload = json.dumps({"chunk": chunk})
                        yield "data: " + payload + "\n\n"
                    sources = []
                    for d in docs:
                        sources.append({
                            "text": d.text[:200],
                            "score": d.score,
                            "source": d.source,
                        })
                    payload = json.dumps({"done": True, "sources": sources})
                    yield "data: " + payload + "\n\n"
                except Exception as exc:
                    error_text = str(exc)
                    if _is_quota_error(error_text):
                        header = "**LLM temporarily unavailable.** Showing raw retrieved chunks for: '" + query + "'"
                        parts = [header]
                        for i, doc in enumerate(docs):
                            idx = str(i + 1)
                            score_str = str(round(doc.score, 3))
                            parts.append("[Source " + idx + "] (score: " + score_str + ") " + doc.text)
                        fallback = "\n\n".join(parts)
                        payload = json.dumps({"chunk": fallback, "done": True, "sources": []})
                        yield "data: " + payload + "\n\n"
                    else:
                        payload = json.dumps({"error": error_text})
                        yield "data: " + payload + "\n\n"
            return Response(event_stream(), mimetype="text/event-stream", content_type="text/event-stream")

        try:
            answer = generate_answer(messages)
        except Exception as exc:
            error_text = str(exc)
            if _is_quota_error(error_text):
                header = "**LLM temporarily unavailable.** Showing raw retrieved chunks for: '" + query + "'"
                parts = [header]
                for i, doc in enumerate(docs):
                    idx = str(i + 1)
                    score_str = str(round(doc.score, 3))
                    parts.append("[Source " + idx + "] (score: " + score_str + ") " + doc.text)
                answer = "\n\n".join(parts)
            else:
                raise

        prompt_tokens = _estimate_tokens(context + query)
        completion_tokens = _estimate_tokens(answer)
        latency = round((time.time() - start_time) * 1000, 2)
        sources = []
        for d in docs:
            sources.append({
                "text": d.text[:200],
                "score": d.score,
                "source": d.source,
            })
        return jsonify({
            "answer": answer,
            "sources": sources,
            "tokens_used": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
            },
            "latency_ms": latency,
        }), 200

    except Exception as exc:
        logger.error("query_failed", error=str(exc), exc_info=True)
        latency = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            "error": "Internal server error during query: " + str(exc),
            "latency_ms": latency,
        }), 500
