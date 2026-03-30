# RAG Stock Market Intelligence

Production-oriented hybrid RAG system for stock-market and financial-news question answering with citation grounding.

## Overview

This project combines:
- FastAPI backend for serving the AI pipeline
- Streamlit frontend for interactive querying
- LangGraph orchestration for multi-step reasoning
- ChromaDB for vector retrieval
- BM25 for keyword retrieval
- Cross-encoder reranking
- Google Gemini for guardrails, routing, query rewriting, and final answer generation

The system ingests a financial-news dataset, builds local retrieval indexes, and answers questions using retrieved context plus inline citations.

## End-to-End Flow

1. User asks a question in the Streamlit UI or through the API.
2. The LangGraph pipeline validates the query for safety and topic relevance.
3. The query is optionally rewritten for retrieval.
4. Gemini classifies intent such as `news`, `reports`, or `general`.
5. Hybrid retrieval runs over ChromaDB and BM25.
6. Retrieved chunks are checked for relevance.
7. A cross-encoder reranks the best chunks.
8. Gemini generates a grounded answer using only retrieved context.
9. The response is returned with citations and latency metadata.

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Keyword Search | BM25 |
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
|-- scripts/
|   |-- ingest.py
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
|   |-- keyword_search.py
|   |-- list_models.py
|   |-- models.py
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
- A Gemini API key with access to Gemini text-generation models

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone <your-repo-url>
cd RAG_System
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Configure environment variables

Create a local `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Set:

```env
GEMINI_API_KEY=your_key_here
```

`GOOGLE_API_KEY` is also supported as a backward-compatible alias, but `GEMINI_API_KEY` is the preferred variable.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify Gemini model access

```powershell
python -m src.list_models
```

## Data Ingestion

Run ingestion once to build the local ChromaDB and BM25 indexes:

```powershell
python -m scripts.ingest
```

Expected output:
- local ChromaDB under `data/chroma_db`
- BM25 index under `data/bm25_index.pkl`

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

## Configuration

Main runtime settings live in [config/settings.yaml](config/settings.yaml).

Important sections:
- `generator`: Gemini model, fallback models, token limits, thinking budget, context trimming
- `retrieval`: vector and BM25 merge behavior
- `reranker`: cross-encoder config
- `embedding`: embedding backend and device
- `agent`: retry and relevance thresholds

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

## Notes for Development

- Local runtime data is stored in `data/`
- Virtual environments and local secrets are ignored through `.gitignore`
- The project does not require Docker for local development
- `docker-compose.yml` is optional and mainly useful if you want to run the API and UI as containers

## Common Commands

```powershell
python -m src.list_models
python -m scripts.ingest
python -m uvicorn api.main:app --reload
python -m streamlit run ui/app.py
python -m src.evaluation
```
