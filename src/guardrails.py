"""
Safety Guardrails
- Topic relevance check (is the query about finance/stocks?)
- Query length validation
- PII blocking
- Relevance assessment of retrieved context
"""

from __future__ import annotations

import re
from typing import Tuple

from loguru import logger

from src.config import settings
from src.gemini_client import generate_text


# ── PII patterns ─────────────────────────────────────────────
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",         # SSN
    r"\b\d{16}\b",                      # Credit card (simple)
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{10}\b",                      # Phone (10 digits)
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    """
    Validate a user query.
    Returns (is_safe, reason).
    """
    cfg = settings.guardrails

    # Length check
    if len(query) > cfg.max_query_length:
        return False, f"Query too long ({len(query)} > {cfg.max_query_length} chars)"

    if len(query.strip()) == 0:
        return False, "Empty query"

    # PII check
    if cfg.block_pii:
        for pattern in PII_PATTERNS:
            if re.search(pattern, query):
                return False, "Query contains potential PII — blocked for safety"

    return True, "ok"


def check_topic_relevance(query: str) -> Tuple[bool, str]:
    """
    Use a lightweight LLM call to determine if the query is
    about finance/stock market topics.
    """
    cfg = settings.guardrails
    allowed = ", ".join(cfg.allowed_topics)

    decision = generate_text(
        query,
        system_instruction=(
            "You are a topic classifier. Determine if the user's query is "
            f"related to any of these topics: {allowed}. "
            "Respond with exactly 'YES' or 'NO'."
        ),
        temperature=0.0,
        max_output_tokens=10,
        thinking_budget=0,
    ).strip().upper()

    if "YES" in decision:
        return True, "Topic is relevant"
    else:
        return False, (
            "Your question doesn't appear to be about stock market or finance. "
            "Please ask a question related to financial markets."
        )


def check_context_relevance(
    query: str, context_snippets: list[str]
) -> Tuple[bool, float]:
    """
    Assess whether retrieved context is relevant to the query.
    Returns (is_relevant, confidence_score).
    """
    if not context_snippets:
        return False, 0.0

    combined = "\n---\n".join(context_snippets[:3])  # Check top 3

    try:
        score = float(
            generate_text(
                f"Query: {query}\n\nContext:\n{combined}",
                system_instruction=(
                    "Rate how relevant the following context is to the query. "
                    "Respond with a single number from 0.0 to 1.0."
                ),
                temperature=0.0,
                max_output_tokens=10,
                thinking_budget=0,
            ).strip()
        )
    except ValueError:
        score = 0.5  # Default if parsing fails

    threshold = settings.agent.relevance_threshold
    is_relevant = score >= threshold

    logger.info(f"🛡️ Context relevance: {score:.2f} (threshold={threshold})")
    return is_relevant, score
