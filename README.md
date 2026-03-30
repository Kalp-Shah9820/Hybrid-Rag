# RAG Stock Market Intelligence

Production-oriented hybrid RAG system for stock-market and financial-news question answering with citation grounding.

## Overview

This version uses a PostgreSQL-native hybrid retrieval stack:
- FastAPI backend for serving the AI pipeline
- Streamlit frontend for interactive querying
- LangGraph orchestration for multi-step reasoning
- PostgreSQL + `pgvector` for semantic retrieval
- PostgreSQL full-text search with `tsvector` + GIN for keyword retrieval
- Deterministic NLP query-understanding layer for query cleanup, synonym expansion, and ticker normalization
- Cross-encoder reranking
- Google Gemini for guardrails, routing, query rewriting, and final answer generation

The system ingests chunked financial-news documents, stores them in PostgreSQL, and retrieves context using a weighted hybrid score that combines semantic and keyword signals.

## End-to-End Flow

1. User asks a question in the Streamlit UI or through the API.
2. The LangGraph pipeline validates the query for safety and topic relevance.
3. Gemini optionally rewrites the query for retrieval.
4. Gemini classifies intent such as `news`, `reports`, or `general`.
5. The NLP layer normalizes the query, removes low-signal tokens, expands finance synonyms, and maps company names to common ticker symbols.
6. The system generates a query embedding from the semantic query.
7. PostgreSQL runs semantic search and full-text search in one hybrid SQL pipeline.
8. Scores are normalized and combined using weighted hybrid ranking.
9. Retrieved chunks are checked for relevance.
10. A cross-encoder reranks the best chunks.
11. Gemini generates a grounded answer using only retrieved context.
12. The response is returned with citations and latency metadata.

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| Semantic Search | PostgreSQL + `pgvector` |
| Keyword Search | PostgreSQL full-text search |
| Embeddings | `all-MiniLM-L6-v2` or Cohere |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini 2.5 Flash with fallback models |
| Evaluation | RAGAS |

## Project Structure

```text
RAG_System/
|-- api/
|   |-- main.py
|-- config/
|   |-- settings.yaml
|-- docs/
|   |-- postgres_hybrid_rag.md
|-- scripts/
|   |-- ingest.py
|   |-- migrate_chroma_to_postgres.py
|-- sql/
|   |-- postgres_pgvector_schema.sql
|-- src/
|   |-- config.py
|   |-- database.py
|   |-- embeddings.py
|   |-- evaluation.py
|   |-- gemini_client.py
|   |-- generator.py
|   |-- graph.py
|   |-- guardrails.py
|   |-- ingestion.py
|   |-- migration.py
|   |-- models.py
|   |-- query_understanding.py
|   |-- reranker.py
|   |-- retriever.py
|-- tests/
|   |-- test_retriever.py
|-- ui/
|   |-- app.py
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
```

## Prerequisites

- Python 3.11+
- Docker Desktop if you want the easiest local PostgreSQL setup
- A Gemini API key with access to Gemini text-generation models

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone <your-repo-url>
cd RAG_System_new
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Configure environment variables

Create a local `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Set at minimum:

```env
GEMINI_API_KEY=your_key_here
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=rag_stockmarket
DATABASE_USER=rag_user
DATABASE_PASSWORD=rag_password
DATABASE_SSLMODE=prefer
```

`GOOGLE_API_KEY` is also supported as a backward-compatible alias, but `GEMINI_API_KEY` is preferred.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start PostgreSQL with pgvector

```powershell
docker compose up -d postgres
```

This uses the `ankane/pgvector` image defined in `docker-compose.yml`.

### 5. Verify Gemini model access

```powershell
python -m src.list_models
```

## Data Setup

You have two supported paths.

### Option A: Fresh ingestion from the dataset

```powershell
python -m scripts.ingest
```

This will:
- create the PostgreSQL schema
- chunk the dataset
- generate embeddings
- insert chunks into PostgreSQL
- build the configured `pgvector` index

### Option B: Migrate existing Chroma data into PostgreSQL

If you already have chunk text, metadata, and embeddings in `data/chroma_db`, run:

```powershell
python -m scripts.migrate_chroma_to_postgres
```

This preserves:
- chunk `id`
- `content`
- `metadata`
- stored embeddings

## Run the Backend

```powershell
python -m uvicorn api.main:app --reload
```

Backend endpoints:
- `POST /ask`
- `GET /health`
- `GET /stats`

Base URL:
- `http://localhost:8000`

## Run the Frontend

In a second terminal with the same environment activated:

```powershell
python -m streamlit run ui/app.py
```

Frontend URL:
- `http://localhost:8501`

## Retrieval Architecture

The hybrid retrieval stack now works as follows:

### Storage

- Each chunk is stored in PostgreSQL in the `document_chunks` table.
- Embeddings are stored in a `vector` column.
- Metadata is stored in `JSONB`.
- A generated `tsvector` column powers keyword search.

### Query Understanding

Before retrieval, the system:
- normalizes whitespace and punctuation
- lowercases and tokenizes the query
- removes low-signal stopwords
- expands finance-specific synonyms such as `earnings -> profit, results, guidance`
- normalizes common company names to ticker symbols such as `tesla -> TSLA`

This improves retrieval quality by helping:
- semantic search understand paraphrased intent
- keyword search match exact entities and alternative terms
- hybrid search recover relevant chunks even when the user query is short, noisy, or phrased differently from the source text

### Hybrid Ranking

PostgreSQL computes:
- semantic similarity from `pgvector`
- keyword relevance from `ts_rank_cd`

Both scores are normalized per candidate set and combined as:

```text
hybrid_score = alpha * semantic_score + beta * keyword_score
```

Default weights:
- `alpha = 0.65`
- `beta = 0.35`

General tuning guidance:
- increase `alpha` for conceptual or paraphrased queries
- increase `beta` for exact-match workloads such as tickers, regulation names, dates, or quoted phrases

More details are documented in `docs/postgres_hybrid_rag.md`.

## Configuration

Main runtime settings live in `config/settings.yaml`.

Important sections:
- `database`: PostgreSQL connection and schema SQL path
- `vector_db`: table name, distance metric, and vector index type
- `retrieval`: semantic top-k, keyword top-k, final top-k, and hybrid weights
- `query_understanding`: synonym expansion and ticker normalization
- `reranker`: cross-encoder config
- `embedding`: embedding backend and device
- `agent`: retry and relevance thresholds
- `generator`: Gemini model, fallback models, token limits, and context trimming

## Gemini Model Strategy

The project uses:
- primary model: `gemini-2.5-flash`
- fallback models:
  - `gemini-2.5-flash-lite`
  - `gemini-2.0-flash`

At startup, the backend validates model availability and uses the first visible supported model from the configured list.

## Evaluation

Run evaluation with:

```powershell
python -m src.evaluation
```

This generates evaluation results in:

```text
data/eval_results.json
```

## Common Commands

```powershell
python -m pip install -r requirements.txt
docker compose up -d postgres
python -m scripts.ingest
python -m scripts.migrate_chroma_to_postgres
python -m uvicorn api.main:app --reload
python -m streamlit run ui/app.py
python -m src.list_models
python -m src.evaluation
```

## Notes

- PostgreSQL replaces the previous ChromaDB + local BM25 artifact split.
- Full-text search is handled inside PostgreSQL, so there is no separate BM25 index file to maintain.
- `docker-compose.yml` now includes a `pgvector`-enabled PostgreSQL service.
- If you use `ivfflat` instead of `hnsw`, build the vector index after bulk ingestion for best results.
