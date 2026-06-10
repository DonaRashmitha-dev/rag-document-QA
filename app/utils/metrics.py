# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""Prometheus metrics for latency, token usage, and retrieval quality."""

from prometheus_client import Counter, Gauge, Histogram

INGEST_REQUESTS = Counter(
    "rag_ingest_requests_total",
    "Total number of document ingest requests",
    ["status"],
)

QUERY_REQUESTS = Counter(
    "rag_query_requests_total",
    "Total number of query requests",
    ["status"],
)

INGEST_CHUNKS = Counter(
    "rag_ingest_chunks_total",
    "Total number of chunks indexed",
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query end-to-end latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

INGEST_LATENCY = Histogram(
    "rag_ingest_latency_seconds",
    "Ingest end-to-end latency in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

TOKENS_USED = Counter(
    "rag_tokens_used_total",
    "Total LLM tokens consumed",
    ["type"],
)

RETRIEVAL_HITS = Counter(
    "rag_retrieval_hits_total",
    "Chunks returned above score threshold",
)

RETRIEVAL_MISSES = Counter(
    "rag_retrieval_misses_total",
    "Queries with no chunks above score threshold",
)

FAISS_CHUNK_COUNT = Gauge(
    "rag_faiss_chunks",
    "Current number of chunks in the FAISS index",
)

UPTIME_SECONDS = Gauge(
    "rag_uptime_seconds",
    "Application uptime in seconds",
)
