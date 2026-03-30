"""
Pydantic data models used across the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of a source document."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str = ""
    title: str = ""
    content: str = ""
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """Chunk returned by the retriever with relevance scores."""
    chunk: DocumentChunk
    vector_score: float = 0.0
    fts_score: float = 0.0
    combined_score: float = 0.0
    raw_vector_score: float = 0.0
    raw_fts_score: float = 0.0
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    """A citation reference in the generated answer."""
    source_index: int
    title: str = ""
    snippet: str = ""
    doc_id: str = ""


class GeneratedAnswer(BaseModel):
    """The final answer returned to the user."""
    query: str
    answer: str
    citations: List[Citation] = []
    chunks_used: List[RetrievedChunk] = []
    is_grounded: bool = True
    latency_ms: float = 0.0


class QueryInput(BaseModel):
    """User query payload for the API."""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    db_connected: bool = False
