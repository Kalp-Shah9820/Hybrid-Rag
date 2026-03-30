"""
Keyword Search Layer — BM25 implementation.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from loguru import logger

from src.config import settings
from src.models import DocumentChunk


class BM25Searcher:
    """Simple BM25 searcher using rank_bm25."""
    
    _instance: Optional[BM25Searcher] = None

    def __init__(self, chunks: List[DocumentChunk] = None):
        self.persist_path = Path(settings.keyword_search.persist_path)
        self.k1 = settings.keyword_search.k1
        self.b = settings.keyword_search.b
        
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[DocumentChunk] = []

        if chunks:
            self._fit(chunks)
        elif self.persist_path.exists():
            self._load()

    def _fit(self, chunks: List[DocumentChunk]):
        """Build the BM25 index from chunks."""
        tokenized_corpus = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.corpus = chunks
        self._save()
        logger.info(f"✅ BM25 index built with {len(chunks)} documents")

    def _save(self):
        """Persist index and corpus."""
        os.makedirs(self.persist_path.parent, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus}, f)

    def _load(self):
        """Load from disk."""
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]
        logger.info(f"✅ BM25 index loaded from {self.persist_path}")

    def search(self, query: str, top_k: int = 20, 
               filter_dict: Optional[dict] = None) -> List[dict]:
        """Search using BM25 with optional filtering."""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        
        # If no filter, use full corpus
        if not filter_dict:
            scores = self.bm25.get_scores(tokenized_query)
            corpus_subset = self.corpus
        else:
            # Filter corpus first (slow for large scale, but fine for in-memory)
            # filter_dict is simple {key: value} mapping
            indices = []
            corpus_subset = []
            for i, chunk in enumerate(self.corpus):
                match = True
                for k, v in filter_dict.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    indices.append(i)
                    corpus_subset.append(chunk)
            
            if not corpus_subset:
                return []
                
            # Recalculate BM25 on subset or just take scores from full index
            # Taking scores from full index is faster and keeps IDF consistency
            all_scores = self.bm25.get_scores(tokenized_query)
            scores = [all_scores[i] for i in indices]
        
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = corpus_subset[idx]

            results.append({
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "score": float(score)
            })
            
        # Sort and take top k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @classmethod
    def get_instance(cls, force_rebuild: bool = False, chunks: List[DocumentChunk] = None) -> BM25Searcher:
        if cls._instance is None or force_rebuild:
            cls._instance = BM25Searcher(chunks=chunks)
        return cls._instance


# API for retriever module
def bm25_search(query: str, top_k: int = 20, filter_dict: Optional[dict] = None) -> List[dict]:
    return BM25Searcher.get_instance().search(query, top_k, filter_dict)


def update_bm25_index(chunks: List[DocumentChunk]):
    BM25Searcher.get_instance(force_rebuild=True, chunks=chunks)
