"""
Deterministic NLP layer for hybrid retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.config import settings


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "when",
    "where", "which", "who", "why", "with",
}

FINANCE_SYNONYMS = {
    "stock": ["equity", "share"],
    "stocks": ["equity", "shares"],
    "earnings": ["profit", "results", "guidance"],
    "revenue": ["sales", "turnover"],
    "fed": ["federal reserve"],
    "inflation": ["cpi", "prices"],
    "bullish": ["upside", "positive"],
    "bearish": ["downside", "negative"],
    "guidance": ["outlook", "forecast"],
    "acquisition": ["takeover", "merger"],
}

TICKER_ALIASES = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "amazon": "AMZN",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "nvidia": "NVDA",
}


def _extract_entities(query: str) -> List[str]:
    """Extract potential stock tickers and company names."""
    entities = []
    # Extract uppercase tickers like AAPL, TSLA
    ticker_pattern = r'\b[A-Z]{2,5}\b'
    tickers = re.findall(ticker_pattern, query)
    entities.extend(tickers)
    
    # Extract known aliases
    words = query.lower().split()
    for word in words:
        if word in TICKER_ALIASES:
            entities.append(TICKER_ALIASES[word])
    
    return list(set(entities))  # Remove duplicates


@dataclass
class QueryBundle:
    original_query: str
    cleaned_query: str
    semantic_query: str
    keyword_query: str
    tokens: List[str]
    expanded_terms: List[str]
    entities: List[str]  # Extracted entities like stock names, tickers


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_query(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9$.\-\s]", " ", text)
    return _normalize_whitespace(text)


def _extract_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9$.\-]+", text.lower())


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    if settings.query_understanding.enable_ticker_normalization and token in TICKER_ALIASES:
        return TICKER_ALIASES[token].lower()
    return token


def _expand_terms(tokens: List[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        expanded.extend(FINANCE_SYNONYMS.get(token, []))
        if token.endswith("s") and token[:-1] in FINANCE_SYNONYMS:
            expanded.extend(FINANCE_SYNONYMS[token[:-1]])

    deduped: List[str] = []
    for term in expanded:
        if term not in deduped:
            deduped.append(term)
        if len(deduped) >= settings.query_understanding.max_expansion_terms:
            break
    return deduped


def build_query_bundle(query: str) -> QueryBundle:
    cleaned = _normalize_query(query)
    raw_tokens = _extract_tokens(cleaned)
    tokens = []
    for token in raw_tokens:
        normalized = _normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)

    expanded_terms = _expand_terms(tokens) if settings.query_understanding.enable_synonym_expansion else []

    semantic_parts = [cleaned]
    if expanded_terms:
        semantic_parts.append(" ".join(expanded_terms))
    semantic_query = _normalize_whitespace(" ".join(part for part in semantic_parts if part))

    keyword_terms: List[str] = []
    for token in tokens:
        if token not in keyword_terms:
            keyword_terms.append(token)
    for term in expanded_terms:
        if term not in keyword_terms:
            keyword_terms.append(term)

    keyword_query = " ".join(keyword_terms) if keyword_terms else cleaned

    entities = _extract_entities(query)

    return QueryBundle(
        original_query=query,
        cleaned_query=cleaned,
        semantic_query=semantic_query,
        keyword_query=keyword_query,
        tokens=tokens,
        expanded_terms=expanded_terms,
        entities=entities,
    )
