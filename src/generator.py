"""
LLM Answer Generator — builds a grounded prompt from retrieved chunks,
calls the LLM, and returns an answer with citations.
"""

from __future__ import annotations

import re
import time
from typing import List

from loguru import logger

from src.config import settings
from src.gemini_client import generate_text_with_retry, resolve_generation_model
from src.models import RetrievedChunk, GeneratedAnswer, Citation


def _build_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    cfg = settings.generator
    parts = []
    current_chars = 0
    for i, rc in enumerate(chunks, 1):
        meta_str = ""
        if rc.chunk.title:
            meta_str += f" | Title: {rc.chunk.title}"
        if rc.chunk.metadata.get("date"):
            meta_str += f" | Date: {rc.chunk.metadata['date']}"
        if rc.chunk.metadata.get("source"):
            meta_str += f" | Source: {rc.chunk.metadata['source']}"
        content = rc.chunk.content[: cfg.max_chunk_chars]
        block = f"[Source {i}{meta_str}]\n{content}"
        if parts and current_chars + len(block) > cfg.max_context_chars:
            break
        parts.append(block)
        current_chars += len(block)
    return "\n\n---\n\n".join(parts)


def _extract_citations(
    answer: str, chunks: List[RetrievedChunk]
) -> List[Citation]:
    """Parse [Source N] references from the answer text."""
    pattern = r"\[Source\s+(\d+)\]"
    found = set(int(m) for m in re.findall(pattern, answer))
    citations = []
    for idx in sorted(found):
        if 1 <= idx <= len(chunks):
            rc = chunks[idx - 1]
            citations.append(
                Citation(
                    source_index=idx,
                    title=rc.chunk.title,
                    snippet=rc.chunk.content[:200],
                    doc_id=rc.chunk.doc_id,
                )
            )
    return citations


def generate_answer(
    query: str, chunks: List[RetrievedChunk]
) -> GeneratedAnswer:
    """
    Generate a grounded answer using the LLM.
    - Constructs context from chunks
    - Calls OpenAI LLM
    - Extracts inline citations
    """
    cfg = settings.generator
    t0 = time.perf_counter()

    # Build prompt
    context = _build_context(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"- Answer using ONLY the information in the context above.\n"
        f"- Cite every claim using [Source N] notation.\n"
        f"- If you cannot answer from the context, say so explicitly.\n"
    )

    selected_model = resolve_generation_model()
    logger.info(f"Calling Gemini model '{selected_model}' for answer generation")
    answer_text = generate_text_with_retry(
        user_prompt,
        system_instruction=cfg.system_prompt,
        preferred_model=selected_model,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens,
        retry_max_output_tokens=cfg.retry_max_tokens,
        thinking_budget=cfg.thinking_budget,
    )

    # Extract citations
    citations = _extract_citations(answer_text, chunks)

    latency = (time.perf_counter() - t0) * 1000

    result = GeneratedAnswer(
        query=query,
        answer=answer_text,
        citations=citations,
        chunks_used=chunks,
        is_grounded=len(citations) > 0,
        latency_ms=round(latency, 1),
    )

    logger.info(
        f"✅ Generated answer ({len(answer_text)} chars, "
        f"{len(citations)} citations, {latency:.0f}ms)"
    )
    return result
