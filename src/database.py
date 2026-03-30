"""
PostgreSQL + pgvector database layer.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
from loguru import logger

from src.config import settings
from src.models import DocumentChunk


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _json_filter_clause(filter_dict: Optional[dict]) -> tuple[str, Optional[str]]:
    if not filter_dict:
        return "", None
    return "AND metadata @> %(metadata_filter)s::jsonb", json.dumps(filter_dict)


def _metric_sql() -> tuple[str, str]:
    metric = settings.vector_db.distance_metric
    if metric == "l2":
        return "embedding <-> %(query_embedding)s::vector", "1.0 / (1.0 + (embedding <-> %(query_embedding)s::vector))"
    if metric == "ip":
        return "embedding <#> %(query_embedding)s::vector", "-1.0 * (embedding <#> %(query_embedding)s::vector)"
    return "embedding <=> %(query_embedding)s::vector", "1.0 - (embedding <=> %(query_embedding)s::vector)"


class PostgresDB:
    """Connection-pooled PostgreSQL wrapper for hybrid retrieval."""

    _pool: Optional[SimpleConnectionPool] = None

    @classmethod
    def get_pool(cls) -> SimpleConnectionPool:
        if cls._pool is None:
            cls._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=settings.database.dsn,
            )
        return cls._pool

    @classmethod
    @contextmanager
    def connection(cls) -> Iterator[psycopg2.extensions.connection]:
        pool = cls.get_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            pool.putconn(conn)

    @classmethod
    def init_schema(cls) -> None:
        schema_path = Path(settings.database.schema_path)
        if not schema_path.is_absolute():
            schema_path = Path(__file__).resolve().parent.parent / schema_path

        sql = schema_path.read_text(encoding="utf-8")
        sql = sql.replace("{{EMBEDDING_DIM}}", str(settings.embedding.dimension))
        sql = sql.replace("{{TABLE_NAME}}", settings.vector_db.table_name)
        sql = sql.replace("{{VECTOR_INDEX_NAME}}", f"{settings.vector_db.table_name}_embedding_idx")
        sql = sql.replace("{{GIN_INDEX_NAME}}", f"{settings.vector_db.table_name}_fts_idx")
        sql = sql.replace("{{DOC_CHUNK_INDEX_NAME}}", f"{settings.vector_db.table_name}_doc_chunk_idx")

        with cls.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        logger.info("Database schema initialized")

    @classmethod
    def create_vector_index(cls) -> None:
        table_name = settings.vector_db.table_name
        index_name = f"{table_name}_embedding_idx"
        metric_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }[settings.vector_db.distance_metric]

        if settings.vector_db.index_type == "ivfflat":
            statement = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} USING ivfflat (embedding {metric_ops})
            WITH (lists = %s);
            """
            params: tuple[Any, ...] = (settings.vector_db.ivfflat_lists,)
        else:
            statement = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} USING hnsw (embedding {metric_ops})
            WITH (m = %s, ef_construction = %s);
            """
            params = (
                settings.vector_db.hnsw_m,
                settings.vector_db.hnsw_ef_construction,
            )

        with cls.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, params)
            conn.commit()
        logger.info("Vector index ensured ({})", settings.vector_db.index_type)

    @classmethod
    def insert_chunks(cls, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        table_name = settings.vector_db.table_name
        rows = [
            (
                chunk.id,
                chunk.doc_id,
                chunk.title,
                chunk.content,
                chunk.chunk_index,
                Json(chunk.metadata or {}),
                _vector_literal(chunk.embedding or []),
            )
            for chunk in chunks
        ]

        insert_sql = f"""
        INSERT INTO {table_name} (
            id, doc_id, title, content, chunk_index, metadata, embedding
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            chunk_index = EXCLUDED.chunk_index,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            updated_at = NOW()
        """

        with cls.connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_sql,
                    rows,
                    template="(%s, %s, %s, %s, %s, %s, %s::vector)",
                    page_size=500,
                )
            conn.commit()
        logger.info("Inserted {} chunks into PostgreSQL", len(chunks))

    @classmethod
    def hybrid_search(
        cls,
        query_embedding: List[float],
        keyword_query: str,
        top_k: int,
        semantic_k: int,
        fts_k: int,
        alpha: float,
        beta: float,
        filter_dict: Optional[dict] = None,
    ) -> List[dict]:
        table_name = settings.vector_db.table_name
        filter_clause, metadata_filter = _json_filter_clause(filter_dict)
        order_expression, semantic_score_expression = _metric_sql()

        sql = f"""
        WITH semantic_hits AS (
            SELECT
                id,
                doc_id,
                title,
                content,
                chunk_index,
                metadata,
                {semantic_score_expression} AS raw_semantic_score,
                0.0::double precision AS raw_keyword_score
            FROM {table_name}
            WHERE 1 = 1
            {filter_clause}
            ORDER BY {order_expression}
            LIMIT %(semantic_k)s
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
                ts_rank_cd(
                    search_vector,
                    websearch_to_tsquery(%(language)s, %(keyword_query)s)
                ) AS raw_keyword_score
            FROM {table_name}
            WHERE search_vector @@ websearch_to_tsquery(%(language)s, %(keyword_query)s)
            {filter_clause}
            ORDER BY raw_keyword_score DESC
            LIMIT %(fts_k)s
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
            id,
            doc_id,
            title,
            content,
            chunk_index,
            metadata,
            raw_semantic_score,
            raw_keyword_score,
            semantic_score,
            keyword_score,
            (%(alpha)s * semantic_score) + (%(beta)s * keyword_score) AS hybrid_score
        FROM normalized
        ORDER BY hybrid_score DESC, semantic_score DESC, keyword_score DESC
        LIMIT %(top_k)s
        """

        params = {
            "query_embedding": _vector_literal(query_embedding),
            "keyword_query": keyword_query,
            "language": settings.query_understanding.language,
            "semantic_k": semantic_k,
            "fts_k": fts_k,
            "alpha": alpha,
            "beta": beta,
            "top_k": top_k,
            "metadata_filter": metadata_filter,
        }

        with cls.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    @classmethod
    def get_chunk_count(cls) -> int:
        table_name = settings.vector_db.table_name
        with cls.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
        return int(count)

    @classmethod
    def check_connection(cls) -> bool:
        try:
            with cls.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception as exc:
            logger.warning("Database connection check failed: {}", exc)
            return False


def init_database() -> None:
    PostgresDB.init_schema()


def insert_chunks(chunks: List[DocumentChunk]) -> None:
    PostgresDB.insert_chunks(chunks)


def hybrid_search(
    query_embedding: List[float],
    keyword_query: str,
    top_k: int,
    semantic_k: int,
    fts_k: int,
    alpha: float,
    beta: float,
    filter_dict: Optional[dict] = None,
) -> List[dict]:
    return PostgresDB.hybrid_search(
        query_embedding=query_embedding,
        keyword_query=keyword_query,
        top_k=top_k,
        semantic_k=semantic_k,
        fts_k=fts_k,
        alpha=alpha,
        beta=beta,
        filter_dict=filter_dict,
    )


def get_chunk_count() -> int:
    return PostgresDB.get_chunk_count()


def check_connection() -> bool:
    return PostgresDB.check_connection()


def ensure_vector_index() -> None:
    PostgresDB.create_vector_index()
