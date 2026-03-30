"""
Configuration loader — reads settings.yaml + .env and exposes
a validated Pydantic settings object used everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"


# ── Sub-models ───────────────────────────────────────────────
class VectorDBConfig(BaseModel):
    type: str = "postgres"
    table_name: str = "document_chunks"
    legacy_collection_name: str = "stock_news_chunks"
    index_type: str = "hnsw"
    distance_metric: str = "cosine"
    ivfflat_lists: int = 100
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "rag_stockmarket"
    user: str = "rag_user"
    password: str = "rag_password"
    schema_path: str = "./sql/postgres_pgvector_schema.sql"
    sslmode: str = "prefer"

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.name} "
            f"user={self.user} password={self.password} sslmode={self.sslmode}"
        )



class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    device: str = "cpu"


class ChunkingConfig(BaseModel):
    chunk_size: int = 700
    chunk_overlap: int = 100
    separators: List[str] = ["\n\n", "\n", ". ", " "]


class RetrievalConfig(BaseModel):
    vector_top_k: int = 20
    fts_top_k: int = 20
    final_top_k: int = 5
    vector_weight: float = 0.65
    fts_weight: float = 0.35
    similarity_threshold: float = 0.3
    min_keyword_score: float = 0.0


class QueryUnderstandingConfig(BaseModel):
    language: str = "english"
    enable_synonym_expansion: bool = True
    enable_ticker_normalization: bool = True
    max_expansion_terms: int = 8


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 32
    device: str = "cpu"


class GeneratorConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    fallback_models: List[str] = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]
    temperature: float = 0.1
    max_tokens: int = 1024
    retry_max_tokens: int = 2048
    thinking_budget: int = 0
    max_context_chars: int = 12000
    max_chunk_chars: int = 2500
    timeout_seconds: int = 30
    validate_model_on_startup: bool = True
    system_prompt: str = (
        "You are a financial news analyst. Answer questions using ONLY the "
        "provided context. Cite every claim with [Source N]. If the context "
        'is insufficient, say "I don\'t have enough information."'
    )


class GuardrailsConfig(BaseModel):
    allowed_topics: List[str] = [
        "stock market", "finance", "investing",
        "trading", "economy", "earnings", "market analysis",
    ]
    max_query_length: int = 500
    block_pii: bool = True


class AgentConfig(BaseModel):
    max_retries: int = 2
    relevance_threshold: float = 0.5


class DatasetConfig(BaseModel):
    name: str = "XA7-stock-market-news"
    source: str = "huggingface"
    hf_repo: str = "Lettria/XA7-stock-market-news"
    split: str = "train"


class EvaluationConfig(BaseModel):
    metrics: List[str] = [
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
    ]
    sample_size: int = 50


class AppConfig(BaseModel):
    name: str = "RAG Stock Market Intelligence"
    version: str = "1.0.0"
    log_level: str = "INFO"


# ── Master settings ──────────────────────────────────────────
class Settings(BaseModel):
    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    query_understanding: QueryUnderstandingConfig = QueryUnderstandingConfig()
    reranker: RerankerConfig = RerankerConfig()
    generator: GeneratorConfig = GeneratorConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    agent: AgentConfig = AgentConfig()
    dataset: DatasetConfig = DatasetConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """Load settings from YAML, overlay with env-vars."""
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Normalize Gemini API key naming. Prefer GEMINI_API_KEY, but keep
    # GOOGLE_API_KEY as a backward-compatible fallback for existing setups.
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key

    # Override OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    database = raw.setdefault("database", {})
    database["host"] = os.getenv("DATABASE_HOST", database.get("host", "localhost"))
    database["port"] = int(os.getenv("DATABASE_PORT", database.get("port", 5432)))
    database["name"] = os.getenv("DATABASE_NAME", database.get("name", "rag_stockmarket"))
    database["user"] = os.getenv("DATABASE_USER", database.get("user", "rag_user"))
    database["password"] = os.getenv("DATABASE_PASSWORD", database.get("password", "rag_password"))
    database["sslmode"] = os.getenv("DATABASE_SSLMODE", database.get("sslmode", "prefer"))

    return Settings(**raw)



# Singleton
settings = load_settings()
