"""
Cross-encoder re-ranker — scores query–chunk pairs with a
cross-encoder model and re-orders by relevance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder
from loguru import logger

from src.config import settings
from src.models import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (singleton)."""
    cfg = settings.reranker
    logger.info(f"Loading re-ranker: {cfg.model_name}")
    return CrossEncoder(cfg.model_name, max_length=512, device=cfg.device)


def rerank(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Re-rank retrieved chunks using a cross-encoder.
    Returns top-k sorted by rerank_score descending.
    """
    if not chunks:
        return []

    cfg = settings.reranker
    model = _get_reranker()

    # Build pairs
    pairs = [(query, c.chunk.content) for c in chunks]

    # Score
    scores = model.predict(pairs, batch_size=cfg.batch_size, show_progress_bar=False)

    # Assign scores
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    # Sort descending
    reranked = sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)
    top = reranked[: cfg.top_k]

    logger.info(
        f"🏆 Re-ranked {len(chunks)} → top {len(top)} "
        f"(best score: {top[0].rerank_score:.4f})"
    )
    return top
