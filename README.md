# 🔍 RAG Document Q&A System

> Upload any document. Ask anything. Get answers with sources — powered entirely by local AI, no API keys, no cloud, no cost.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that runs **100% locally** — no OpenAI, no Gemini, no API keys. Ingests documents, indexes them in a FAISS vector store, and exposes a Flask REST API for natural-language Q&A with full source attribution.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | PDF, DOCX, TXT, HTML, CSV |
| **FAISS vector search** | Sub-second exact L2 search with MMR reranking |
| **Local embeddings** | `nomic-embed-text` via Ollama — no cloud calls |
| **Local LLM** | `tinyllama` via Ollama — runs on CPU, no GPU needed |
| **JWT authentication** | Stateless token-based auth on all routes |
| **Rate limiting** | Token-bucket per-IP limiter |
| **Observability** | structlog JSON logging + Prometheus metrics |
| **Docker ready** | Single `docker compose up` deployment |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────────────┐
│  Document   │────▶│  Loader  │────▶│   Chunker   │────▶│    Embedder      │
│ (PDF/DOCX/  │     │          │     │ (512 chars, │     │ nomic-embed-text │
│  TXT/HTML)  │     │          │     │  64 overlap)│     │ via Ollama       │
└─────────────┘     └──────────┘     └─────────────┘     └───────┬──────────┘
                                                                   │
                                                          ┌────────▼────────┐
                                                          │  FAISS Index +  │
                                                          │ Metadata Store  │
                                                          └────────┬────────┘
                                                                   │
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌────────▼────────┐
│   Answer +  │◀────│tinyllama │◀────│   Prompt    │◀────│   Retriever     │
│   Sources   │     │via Ollama│     │   Builder   │     │  (MMR rerank,   │
│             │     │localhost │     │             │     │   top-k=5)      │
└─────────────┘     └──────────┘     └─────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (3.14 works, 3.11/3.12 recommended)
- [Ollama](https://ollama.com) installed and running
- Node.js (only for the React frontend)

### 1. Install Ollama & Pull Models

Download Ollama from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull nomic-embed-text   # embeddings model
ollama pull tinyllama          # LLM (fast, runs on CPU)
```

Verify Ollama is running:
```bash
ollama list
# Should show both models
```

> **Want better answers?** Swap tinyllama for a smarter model:
> ```bash
> ollama pull gemma:2b    # better quality, still lightweight
> ollama pull phi3:mini   # Microsoft's efficient model
> ```
> Then set `OLLAMA_LLM_MODEL=gemma:2b` in your `.env`.

### 2. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/rag_project.git
cd rag_project
cp .env.example .env
```

Open `.env` and set at minimum:
```env
JWT_SECRET=any-random-secret-string-you-choose
```

Everything else has sensible defaults. Ollama runs at `http://localhost:11434` by default.

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Start the Backend

```bash
flask --app app:create_app run --port 8000
```

Verify:
```bash
curl http://127.0.0.1:8000/health
# {"faiss_chunks": 0, "status": "ok", "uptime_s": 2.1}
```

### 5. Start the Frontend (Optional)

```bash
cd rag-frontend
npm install
npm start
# Opens http://localhost:3000 automatically
```

### 6. Docker (All-in-one)

```bash
docker compose up --build
```

---

## 🔑 Getting Your JWT Token

All API routes (except `/health`) require a Bearer token. Generate one:

```bash
# Run from project root with dependencies installed
python -c "from app.middleware.auth import create_access_token; print(create_access_token())"
```

Copy the printed token and use it in all requests:
```
Authorization: Bearer <your-token-here>
```

> Tokens are signed with your `JWT_SECRET`. Changing the secret invalidates existing tokens.

---

## 📡 API Usage

### Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{"chunks_added": 42, "filename": "document.pdf", "status": "ok"}
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "The refund policy allows returns within 30 days...",
  "sources": [
    {"chunk": "Returns are accepted within 30 days...", "score": 0.92, "source": "policy.pdf"}
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health
# {"faiss_chunks": 148, "status": "ok", "uptime_s": 28.9}
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | **required** | Any random string for signing tokens |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `tinyllama` | LLM for answer generation |
| `CHUNK_SIZE` | `512` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.7` | Minimum relevance score |
| `MAX_CONTEXT_TOKENS` | `12000` | Token budget for context |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 💸 Cost

**Completely free. Runs offline. No API keys.**

All inference happens locally via Ollama. Once models are downloaded, zero ongoing cost and zero data leaves your machine.

---

## 📖 API Specification

Full OpenAPI spec in [`openapi.yaml`](./openapi.yaml).

---

## License

MIT — free to use, modify, and distribute.


