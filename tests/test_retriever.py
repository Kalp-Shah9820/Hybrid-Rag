"""
Unit tests for the PostgreSQL-backed retriever module.
"""

from unittest.mock import patch

from src.models import DocumentChunk, RetrievedChunk
from src.query_understanding import QueryBundle


def test_row_to_chunk():
    from src.retriever import _row_to_chunk

    row = {
        "id": "test-id",
        "doc_id": "doc-1",
        "title": "Test Title",
        "content": "Test content",
        "chunk_index": 0,
        "metadata": {"source": "test"},
    }
    chunk = _row_to_chunk(row)
    assert isinstance(chunk, DocumentChunk)
    assert chunk.id == "test-id"
    assert chunk.title == "Test Title"


@patch("src.retriever.hybrid_search")
@patch("src.retriever.embed_query")
@patch("src.retriever.build_query_bundle")
def test_hybrid_retrieve_ranks_results(mock_bundle, mock_embed, mock_hybrid_search):
    from src.retriever import hybrid_retrieve

    mock_bundle.return_value = QueryBundle(
        original_query="test query",
        cleaned_query="test query",
        semantic_query="test query earnings results",
        keyword_query="test query earnings results",
        tokens=["test", "query"],
        expanded_terms=["earnings", "results"],
    )
    mock_embed.return_value = [0.1] * 384
    mock_hybrid_search.return_value = [
        {
            "id": "b",
            "doc_id": "d2",
            "title": "T2",
            "content": "C2",
            "chunk_index": 0,
            "metadata": {},
            "semantic_score": 0.9,
            "keyword_score": 0.7,
            "hybrid_score": 0.83,
            "raw_semantic_score": 0.92,
            "raw_keyword_score": 0.45,
        },
        {
            "id": "a",
            "doc_id": "d1",
            "title": "T1",
            "content": "C1",
            "chunk_index": 0,
            "metadata": {},
            "semantic_score": 0.8,
            "keyword_score": 0.0,
            "hybrid_score": 0.52,
            "raw_semantic_score": 0.81,
            "raw_keyword_score": 0.0,
        },
    ]

    results = hybrid_retrieve("test query")

    assert len(results) == 2
    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert results[0].chunk.id == "b"
    assert results[0].combined_score > results[1].combined_score


@patch("src.retriever.hybrid_search")
@patch("src.retriever.embed_query")
@patch("src.retriever.build_query_bundle")
def test_hybrid_retrieve_empty(mock_bundle, mock_embed, mock_hybrid_search):
    from src.retriever import hybrid_retrieve

    mock_bundle.return_value = QueryBundle(
        original_query="nonexistent query",
        cleaned_query="nonexistent query",
        semantic_query="nonexistent query",
        keyword_query="nonexistent query",
        tokens=["nonexistent", "query"],
        expanded_terms=[],
    )
    mock_embed.return_value = [0.1] * 384
    mock_hybrid_search.return_value = []

    results = hybrid_retrieve("nonexistent query")
    assert results == []
