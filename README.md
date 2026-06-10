# RAG-Powered Document Q&A System



Production-grade Retrieval-Augmented Generation (RAG) pipeline that ingests document corpora, indexes them in a FAISS vector store, and exposes a Flask REST API for natural-language Q&A with source attribution.



## Features



- **Multi-format ingestion** — PDF, DOCX, TXT, HTML, CSV

- **FAISS vector search** — Sub-second exact L2 search with MMR reranking

- **Google Gemini embeddings** — `models/text-embedding-004` (768-dim)

- **Gemini 1.5 Flash answers** — Grounded generation with streaming SSE support

- **JWT authentication** — Stateless token-based auth on all routes except `/health`

- **Rate limiting** — Token-bucket per-IP limiter

- **Observability** — structlog JSON logging + Prometheus metrics



## Free Setup with Google Gemini



This project uses Google's free Gemini API tier — no OpenAI subscription required.



1. **Get a free API key** at [Google AI Studio](https://aistudio.google.com/apikey).

2. **Copy the example env file** and set your key:



   ```bash

   cp .env.example .env

   # Edit .env and set:

   # GOOGLE_API_KEY=your-key-here

   # JWT_SECRET=any-random-secret-string

   ```



3. **Install and run:**



   ```bash

   pip install -e ".[dev]"

   flask --app app:create_app run --port 8000

   ```



Free-tier limits apply (requests per minute/day). See [Gemini API pricing](https://ai.google.dev/pricing) for current quotas.



## Quick Start



```bash

# 1. Clone and install

cp .env.example .env          # fill in GOOGLE_API_KEY and JWT_SECRET

pip install -e ".[dev]"



# 2. Run tests

pytest tests/ -v --cov=app --cov-report=term-missing



# 3. Start server

flask --app app:create_app run --port 8000



# 4. Docker

docker compose up --build

```



## API Usage



Generate a JWT token (for development):



```python

from app.middleware.auth import create_access_token

token = create_access_token()

print(token)

```



### Ingest a document



```bash

curl -X POST http://localhost:8000/ingest \

  -H "Authorization: Bearer <token>" \

  -F "file=@docs/policy.pdf"

```



### Query



```bash

curl -X POST http://localhost:8000/query \

  -H "Authorization: Bearer <token>" \

  -H "Content-Type: application/json" \

  -d '{"query": "What is the refund policy?", "top_k": 5}'

```



### Health check



```bash

curl http://localhost:8000/health

```



## Architecture



```

Ingest:  loader → chunker → embedder → FAISS + metadata store

Query:   retriever (MMR) → prompt builder → ChatGoogleGenerativeAI → response

```



See `openapi.yaml` for the full API specification.



## Configuration



All settings are loaded from environment variables. See `.env.example` for the full list.



| Variable | Default | Description |

|---|---|---|

| `GOOGLE_API_KEY` | — | Required Google Gemini API key |

| `JWT_SECRET` | — | Required JWT signing secret |

| `CHUNK_SIZE` | 512 | Text chunk size in characters |

| `CHUNK_OVERLAP` | 64 | Overlap between chunks |

| `TOP_K` | 5 | Default retrieval count |

| `SCORE_THRESHOLD` | 0.7 | Minimum relevance score |

| `MAX_CONTEXT_TOKENS` | 12000 | Token budget for context |

| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |

| `LLM_MODEL` | `gemini-1.5-flash` | Gemini chat model |



## License



MIT

