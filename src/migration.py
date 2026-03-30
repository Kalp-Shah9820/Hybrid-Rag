"""
Migration utilities for moving ChromaDB data into PostgreSQL + pgvector.
"""

from __future__ import annotations

from typing import List

import chromadb
from loguru import logger

from src.config import settings
from src.database import ensure_vector_index, init_database, insert_chunks
from src.models import DocumentChunk


def _row_to_chunk(
    chunk_id: str,
    content: str,
    metadata: dict,
    embedding: List[float],
) -> DocumentChunk:
    metadata = metadata or {}
    return DocumentChunk(
        id=chunk_id,
        doc_id=metadata.get("doc_id", ""),
        title=metadata.get("title", ""),
        content=content,
        chunk_index=int(metadata.get("chunk_index", 0)),
        metadata=metadata,
        embedding=embedding,
    )


def migrate_chroma_to_postgres(batch_size: int = 500) -> int:
    """Extract existing chunks from ChromaDB and insert them into PostgreSQL."""
    init_database()

    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_collection(settings.vector_db.legacy_collection_name)
    total = collection.count()
    migrated = 0

    logger.info(f"Migrating {total:,} chunks from ChromaDB to PostgreSQL")
    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=batch_size,
            offset=offset,
        )
        chunks = [
            _row_to_chunk(chunk_id, content, metadata, embedding)
            for chunk_id, content, metadata, embedding in zip(
                batch["ids"],
                batch["documents"],
                batch["metadatas"],
                batch["embeddings"],
            )
        ]
        insert_chunks(chunks)
        migrated += len(chunks)
        logger.info(f"Migrated {migrated:,}/{total:,} chunks")

    ensure_vector_index()
    return migrated
