"""
FastAPI Application — REST API for the RAG Stock Market system.

Endpoints:
  POST /ask         — Submit a query and get a grounded answer
  GET  /health      — System health check
  GET  /stats       — Database statistics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import settings
from src.models import QueryInput, GeneratedAnswer, HealthResponse
from src.database import check_connection, get_chunk_count, init_database
from src.gemini_client import validate_gemini_configuration
from src.graph import run_agent


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("🚀 Starting RAG Stock Market API")
    try:
        if settings.generator.provider == "gemini" and settings.generator.validate_model_on_startup:
            active_model = validate_gemini_configuration()
            logger.info(f"✅ Gemini ready with model: {active_model}")

        init_database()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.warning(f"⚠️  Startup initialization skipped: {e}")
    yield
    logger.info("👋 Shutting down")


# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Production-grade Hybrid RAG for Stock Market News",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────

@app.post("/ask", response_model=GeneratedAnswer)
async def ask(payload: QueryInput):
    """
    Submit a stock-market question and receive a grounded answer
    with citations.
    """
    try:
        t0 = time.perf_counter()
        answer = run_agent(payload.query)
        answer.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return answer
    except Exception as e:
        logger.error(f"❌ /ask error: {e}")
        detail = str(e)
        if "quota" in detail.lower() or "rate limit" in detail.lower() or "resource_exhausted" in detail.lower():
            detail = (
                f"⚠️ **Gemini quota error**: {detail}\n\n"
                "Your Gemini API key has hit a quota or rate limit. "
                "Try again later, check your plan, or use a different API key."
            )
        else:
            detail = (
                f"⚠️ **System Error**: {detail}\n\n"
                "The AI pipeline could not complete this request. Check your "
                "Gemini API key, configured model availability, and data index status."
            )
        return GeneratedAnswer(
            query=payload.query,
            answer=detail,
            is_grounded=False,
            latency_ms=0.0
        )



@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app.version,
        db_connected=check_connection(),
    )


@app.get("/stats")
async def stats():
    """Return database statistics."""
    try:
        count = get_chunk_count()
        return {
            "total_chunks": count,
            "vector_db_type": settings.vector_db.type,
            "table_name": settings.vector_db.table_name,
            "vector_index_type": settings.vector_db.index_type,
            "embedding_model": settings.embedding.model_name,
            "reranker_model": settings.reranker.model_name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run with: python -m uvicorn api.main:app --reload ────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
