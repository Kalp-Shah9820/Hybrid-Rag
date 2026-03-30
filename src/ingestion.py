"""
Data ingestion pipeline for PostgreSQL + pgvector.
"""

from __future__ import annotations

import hashlib
from typing import List

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import settings
from src.database import ensure_vector_index, get_chunk_count, init_database, insert_chunks
from src.embeddings import get_embeddings
from src.models import DocumentChunk


def load_xa7_dataset() -> list[dict]:
    """Download and return the XA7 stock-market news dataset."""
    cfg = settings.dataset
    logger.info(f"Loading dataset: {cfg.hf_repo} (split={cfg.split})")
    ds = load_dataset(cfg.hf_repo, split=cfg.split)
    records = [dict(row) for row in ds]
    logger.info(f"Loaded {len(records):,} records")
    return records


def _doc_id(text: str) -> str:
    """Deterministic document id from content hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_documents(records: list[dict]) -> List[DocumentChunk]:
    """Split raw records into overlapping chunks with robust field mapping."""
    cfg = settings.chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )

    chunks: List[DocumentChunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=len(records))

        for record in records:
            def get_field(keys_to_try: list[str]):
                for key in record.keys():
                    if key.lower() in keys_to_try:
                        return record[key]
                return None

            title = get_field(["headline", "title", "name", "description"]) or ""
            text = get_field(["text", "content", "body"]) or title

            if not text:
                progress.advance(task)
                continue

            source = get_field(["source"]) or "news"
            date = get_field(["date", "time"]) or ""
            label = get_field(["label", "target"]) or ""
            category = get_field(["category", "genre"]) or ""
            doc_id = _doc_id(str(text))

            for idx, split_text in enumerate(splitter.split_text(str(text))):
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        title=str(title),
                        content=str(split_text),
                        chunk_index=idx,
                        metadata={
                            "source": str(source),
                            "date": str(date),
                            "url": str(get_field(["url", "link"]) or ""),
                            "category": str(category),
                            "label": str(label),
                        },
                    )
                )

            progress.advance(task)

    logger.info(f"Created {len(chunks):,} chunks from {len(records):,} documents")
    return chunks


def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Generate embeddings for all chunks in batches."""
    embedder = get_embeddings()
    texts = [chunk.content for chunk in chunks]
    batch_size = settings.embedding.batch_size
    all_embeddings = []

    logger.info(f"Generating embeddings (batch_size={batch_size})")
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        all_embeddings.extend(embedder.embed_documents(batch))

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk.embedding = embedding

    logger.info(f"Embedded {len(chunks):,} chunks")
    return chunks


def run_ingestion() -> None:
    """End-to-end ingestion: load -> chunk -> embed -> store."""
    logger.info("Starting ingestion pipeline")
    init_database()

    existing = get_chunk_count()
    if existing > 0:
        logger.warning(
            f"Database already contains {existing:,} chunks. "
            "Skipping ingestion. Truncate the table to re-ingest."
        )
        return

    chunks = chunk_documents(load_xa7_dataset())
    if not chunks:
        logger.warning("No chunks created. Ingestion stopped.")
        return

    insert_chunks(embed_chunks(chunks))
    ensure_vector_index()
    total = get_chunk_count()
    logger.info(f"Ingestion complete: {total:,} chunks in PostgreSQL")


if __name__ == "__main__":
    run_ingestion()
