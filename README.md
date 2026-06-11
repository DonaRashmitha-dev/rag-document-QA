# RAG Document Q&A System

> Upload any document. Ask anything. Get answers with sources — powered entirely by local AI, no API keys, no cloud, no cost.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that runs **100% locally** — no OpenAI, no Gemini, no API keys. Ingests documents, indexes them in a FAISS vector store, and exposes a Flask REST API for natural-language Q&A with full source attribution.

---

## Screenshots

![Document Ingestion](screenshot%201.png)
![RAG Answer](screenshot%202.png)

---

## Features

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

## Architecture

```
Document → Loader → Chunker → Embedder (nomic-embed-text) → FAISS Index
                                                                   |
Answer + Sources ← tinyllama ← Prompt Builder ← Retriever (MMR, top-k=5)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Node.js (only for the React frontend)

### 1. Install Ollama & Pull Models

```bash
ollama pull nomic-embed-text
ollama pull tinyllama
```

Want better answers? Use a smarter model:
```bash
ollama pull phi3:mini
# Then set ollama_llm_model = "phi3:mini" in app/core/config.py
```

### 2. Clone & Configure

```bash
git clone https://github.com/DonaRashmitha-dev/rag-document-QA.git
cd rag-document-QA
cp .env.example .env
```

Open `.env` and set:
```env
JWT_SECRET=any-random-secret-string
```

### 3. Install & Run

```bash
pip install -e ".[dev]"
flask --app app:create_app run --port 8000
```

### 4. Frontend (Optional)

```bash
cd rag-frontend
npm install
npm start
```

### 5. Docker

```bash
docker compose up --build
```

---

## Getting Your JWT Token

```bash
python -c "
import jwt
from datetime import datetime, timedelta, timezone
payload = {
    'sub': 'rag-qa-client',
    'iat': datetime.now(timezone.utc),
    'exp': datetime.now(timezone.utc) + timedelta(hours=24)
}
print(jwt.encode(payload, 'your-jwt-secret', algorithm='HS256'))
"
```

Use the printed token as: `Authorization: Bearer <token>`

---

## API Usage

### Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | **required** | Any random string for signing tokens |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `tinyllama` | LLM for answer generation |
| `CHUNK_SIZE` | `512` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Cost

**Completely free. Runs offline. No API keys.**

All inference happens locally via Ollama. Zero ongoing cost, zero data leaves your machine.

---

## License

MIT
