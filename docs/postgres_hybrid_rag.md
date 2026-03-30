# PostgreSQL + pgvector Hybrid RAG

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
        setweight(jsonb_to_tsvector('english', metadata, '["string"]'), 'C')
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX document_chunks_fts_idx
ON document_chunks USING GIN (search_vector);

CREATE INDEX document_chunks_embedding_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Hybrid Query

```sql
WITH semantic_hits AS (
    SELECT
        id,
        doc_id,
        title,
        content,
        chunk_index,
        metadata,
        1 - (embedding <=> $1::vector) AS raw_semantic_score,
        0.0::double precision AS raw_keyword_score
    FROM document_chunks
    ORDER BY embedding <=> $1::vector
    LIMIT $2
),
keyword_hits AS (
    SELECT
        id,
        doc_id,
        title,
        content,
        chunk_index,
        metadata,
        0.0::double precision AS raw_semantic_score,
        ts_rank_cd(search_vector, websearch_to_tsquery('english', $3)) AS raw_keyword_score
    FROM document_chunks
    WHERE search_vector @@ websearch_to_tsquery('english', $3)
    ORDER BY raw_keyword_score DESC
    LIMIT $4
),
candidates AS (
    SELECT * FROM semantic_hits
    UNION ALL
    SELECT * FROM keyword_hits
),
deduped AS (
    SELECT
        id,
        MIN(doc_id) AS doc_id,
        MIN(title) AS title,
        MIN(content) AS content,
        MAX(chunk_index) AS chunk_index,
        (ARRAY_AGG(metadata))[1] AS metadata,
        MAX(raw_semantic_score) AS raw_semantic_score,
        MAX(raw_keyword_score) AS raw_keyword_score
    FROM candidates
    GROUP BY id
),
normalized AS (
    SELECT
        *,
        CASE
            WHEN MAX(raw_semantic_score) OVER () > 0
            THEN raw_semantic_score / MAX(raw_semantic_score) OVER ()
            ELSE 0.0
        END AS semantic_score,
        CASE
            WHEN MAX(raw_keyword_score) OVER () > 0
            THEN raw_keyword_score / MAX(raw_keyword_score) OVER ()
            ELSE 0.0
        END AS keyword_score
    FROM deduped
)
SELECT
    *,
    (0.65 * semantic_score) + (0.35 * keyword_score) AS hybrid_score
FROM normalized
ORDER BY hybrid_score DESC
LIMIT $5;
```

## Query Understanding

- Normalize punctuation, whitespace, and case.
- Remove low-signal stopwords before keyword retrieval.
- Expand finance terms with domain synonyms to improve recall.
- Normalize common company names into ticker symbols so semantic and keyword search align.
- Use `semantic_query` for embeddings and `keyword_query` for `websearch_to_tsquery`.

## Ranking Guidance

- Raise `vector_weight` when users ask conceptual or paraphrased questions.
- Raise `fts_weight` when users use exact terms, ticker symbols, dates, or regulation names.
- Start with `0.65 / 0.35` and tune from clickthrough or answer-grounding quality.

## Migration

```bash
python -m scripts.migrate_chroma_to_postgres
```

This reads legacy Chroma chunks from `data/chroma_db`, preserves `id`, `content`, `metadata`, and embeddings, and upserts them into PostgreSQL.
