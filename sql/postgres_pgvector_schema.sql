CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {{TABLE_NAME}} (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(content, '')), 'B') ||
        setweight(jsonb_to_tsvector('english', metadata, '["string"]'), 'C')
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS {{GIN_INDEX_NAME}}
ON {{TABLE_NAME}} USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS {{DOC_CHUNK_INDEX_NAME}}
ON {{TABLE_NAME}} (doc_id, chunk_index);
