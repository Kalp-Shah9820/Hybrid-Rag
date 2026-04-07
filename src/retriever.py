"""
Hybrid retriever backed by PostgreSQL full-text search and pgvector.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from src.config import settings
from src.database import hybrid_search
from src.embeddings import embed_query
from src.models import DocumentChunk, RetrievedChunk
from src.query_understanding import QueryBundle, build_query_bundle


def _row_to_chunk(row: dict) -> DocumentChunk:
    """Convert a database row dict into a DocumentChunk."""
    return DocumentChunk(
        id=row["id"],
        doc_id=row["doc_id"],
        title=row.get("title", ""),
        content=row["content"],
        chunk_index=row.get("chunk_index", 0),
        metadata=row.get("metadata", {}),
    )


def _build_filter(source_filter: Optional[str]) -> Optional[dict]:
    if not source_filter:
        return None
    return {"source": source_filter}


def _to_retrieved_chunk(row: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_row_to_chunk(row),
        vector_score=float(row.get("semantic_score", 0.0)),
        fts_score=float(row.get("keyword_score", 0.0)),
        combined_score=float(row.get("hybrid_score", 0.0)),
        raw_vector_score=float(row.get("raw_semantic_score", 0.0)),
        raw_fts_score=float(row.get("raw_keyword_score", 0.0)),
    )


def retrieve(query: str, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
    """Run NLP-aware hybrid retrieval and return ranked chunks."""
    cfg = settings.retrieval
    query_bundle: QueryBundle = build_query_bundle(query)
    semantic_query = query_bundle.semantic_query or query_bundle.cleaned_query or query
    query_embedding = embed_query(semantic_query)
    keyword_query = query_bundle.keyword_query or query_bundle.cleaned_query or query
    
    # Boost keyword query with entities
    if query_bundle.entities:
        entity_terms = " ".join(query_bundle.entities)
        keyword_query += " " + entity_terms

    rows = hybrid_search(
        query_embedding=query_embedding,
        keyword_query=keyword_query,
        top_k=cfg.final_top_k,
        semantic_k=cfg.vector_top_k,
        fts_k=cfg.fts_top_k,
        alpha=cfg.vector_weight,
        beta=cfg.fts_weight,
        filter_dict=_build_filter(source_filter),
    )

    results = [_to_retrieved_chunk(row) for row in rows]
    logger.info(
        "Hybrid retrieval returned {} results for query='{}' semantic='{}' keyword='{}' entities={}",
        len(results),
        query,
        semantic_query,
        keyword_query,
        query_bundle.entities,
    )
    return results


def hybrid_retrieve(query: str, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
    return retrieve(query=query, source_filter=source_filter)
