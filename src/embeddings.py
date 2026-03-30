"""
Embedding module — wraps SentenceTransformer via LangChain interface.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from loguru import logger

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings | CohereEmbeddings:
    """Return a singleton embedding model instance."""
    cfg = settings.embedding
    
    if "cohere" in cfg.model_name.lower():
        logger.info(f"Loading Cohere embedding model: {cfg.model_name}")
        return CohereEmbeddings(model=cfg.model_name)
    
    logger.info(f"Loading embedding model: {cfg.model_name} (device={cfg.device})")
    return HuggingFaceEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": cfg.batch_size},
    )



def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return get_embeddings().embed_query(query)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents."""
    return get_embeddings().embed_documents(texts)
