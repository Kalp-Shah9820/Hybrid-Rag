# Run System

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Set these values in `.env`:

```env
GEMINI_API_KEY=your_key_here
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=rag_stockmarket
DATABASE_USER=rag_user
DATABASE_PASSWORD=rag_password
DATABASE_SSLMODE=prefer
```

## Start Infrastructure

Start PostgreSQL with `pgvector`:

```powershell
docker compose up -d postgres
```

## Load Data

Choose one path.

### Fresh ingestion

```powershell
python -m scripts.ingest
```

Use this when you want to build the PostgreSQL index from the dataset.

## Run the Backend

```powershell
python -m uvicorn api.main:app --reload
```

API endpoints:
- `http://localhost:8000/health`
- `http://localhost:8000/stats`
- `http://localhost:8000/ask`

## Run the Frontend

Open a second terminal and activate the same environment:

```powershell
.\venv\Scripts\Activate
python -m streamlit run ui/app.py
```

UI URL:
- `http://localhost:8501`

## Full Docker Flow

If you want PostgreSQL, API, and UI together:

```powershell
docker compose up -d
```

## Sanity Checks

After startup:
- open `http://localhost:8000/health`
- open `http://localhost:8000/stats`
- ask a few queries in the UI

Good test queries:
- `How did Tesla stock perform recently?`
- `What impact did the Fed rate decision have?`
- `What are analysts predicting for tech stocks?`

## Useful Commands

```powershell
python -m src.list_models
python -m scripts.ingest
python -m uvicorn api.main:app --reload
python -m streamlit run ui/app.py
python -m src.evaluation
docker compose up -d postgres
docker compose up -d
```

## Notes

- The current system uses PostgreSQL + `pgvector` for semantic retrieval.
- Keyword retrieval now uses PostgreSQL full-text search, not a separate BM25 file.
- If PostgreSQL already has data, ingestion will skip reloading until you clear the table.
