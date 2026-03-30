"""
LangGraph Agentic Workflow
Implements a state-machine that:
1. Validates the query (guardrails)
2. Rewrites the query if needed
3. Retrieves context (hybrid)
4. Checks context relevance
5. Re-ranks
6. Generates answer
With retry loops for insufficient context.
"""

from __future__ import annotations

from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from loguru import logger

from src.config import settings
from src.gemini_client import generate_text
from src.models import RetrievedChunk, GeneratedAnswer
from src.guardrails import check_query_safety, check_topic_relevance, check_context_relevance
from src.retriever import hybrid_retrieve
from src.reranker import rerank
from src.generator import generate_answer


# ── State definition ─────────────────────────────────────────
class RAGState(TypedDict):
    """State object flowing through the agent graph."""
    original_query: str
    rewritten_query: str
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]
    answer: Optional[GeneratedAnswer]
    is_safe: bool
    is_relevant_topic: bool
    is_relevant_context: bool
    context_relevance_score: float
    retry_count: int
    error_message: str
    status: str  # "processing" | "success" | "blocked" | "error"
    intent: str   # "news" | "reports" | "general"
    source_filter: Optional[str]



# ── Node functions ───────────────────────────────────────────

def validate_query(state: RAGState) -> RAGState:
    """Check query safety and topic relevance."""
    query = state["original_query"]
    logger.info(f"🔒 Validating query: {query[:80]}...")

    # Safety check
    is_safe, reason = check_query_safety(query)
    if not is_safe:
        logger.warning(f"⛔ Query blocked: {reason}")
        return {
            **state,
            "is_safe": False,
            "error_message": reason,
            "status": "blocked",
        }

    # Topic check
    is_relevant, reason = check_topic_relevance(query)
    if not is_relevant:
        logger.warning(f"⛔ Off-topic query: {reason}")
        return {
            **state,
            "is_safe": True,
            "is_relevant_topic": False,
            "error_message": reason,
            "status": "blocked",
        }

    return {
        **state,
        "is_safe": True,
        "is_relevant_topic": True,
        "status": "processing",
    }


def rewrite_query(state: RAGState) -> RAGState:
    """Optionally rewrite the query for better retrieval."""
    query = state["original_query"]
    retry = state.get("retry_count", 0)

    # On first pass, try a light rewrite; on retries, rephrase more aggressively
    if retry == 0:
        prompt = (
            "Rewrite the following user query to improve search retrieval for "
            "a stock market news database. Keep it concise. Return ONLY the "
            "rewritten query, nothing else."
        )
    else:
        prompt = (
            "The previous search did not find relevant results. Aggressively "
            "rephrase this query using different keywords, synonyms, and "
            "broader terms related to stock market and finance. Return ONLY "
            "the rewritten query."
        )

    rewritten = generate_text(
        query,
        system_instruction=prompt,
        temperature=0.3,
        max_output_tokens=100,
        thinking_budget=0,
    )
    logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
    return {**state, "rewritten_query": rewritten}

def route_query(state: RAGState) -> RAGState:
    """Classify query intent to route to correct data source."""
    query = state.get("rewritten_query") or state["original_query"]
    
    prompt = (
        "Classify the following stock market question into one of three "
        "categories: 'news', 'reports', or 'general'. Return ONLY the single "
        "word of the category."
    )
    
    intent = generate_text(
        query,
        system_instruction=prompt,
        temperature=0.0,
        max_output_tokens=20,
        thinking_budget=0,
    ).strip().lower()
    
    # Simple mapping
    if intent not in ["news", "reports", "general"]:
        intent = "news" # Fallback
        
    logger.info(f"🛣️ Query routed: category='{intent}'")
    return {**state, "intent": intent, "source_filter": intent if intent != "general" else None}


def retrieve(state: RAGState) -> RAGState:
    """Run hybrid retrieval with source routing."""
    query = state.get("rewritten_query") or state["original_query"]
    source = state.get("source_filter")
    logger.info(f"🔍 Retrieving for: {query[:80]}... (filter: {source})")
    
    chunks = hybrid_retrieve(query, source_filter=source)
    return {**state, "retrieved_chunks": chunks}



def check_relevance(state: RAGState) -> RAGState:
    """Check if retrieved context is relevant enough."""
    chunks = state.get("retrieved_chunks", [])
    query = state.get("rewritten_query") or state["original_query"]

    if not chunks:
        return {
            **state,
            "is_relevant_context": False,
            "context_relevance_score": 0.0,
        }

    snippets = [c.chunk.content for c in chunks[:3]]
    is_relevant, score = check_context_relevance(query, snippets)

    return {
        **state,
        "is_relevant_context": is_relevant,
        "context_relevance_score": score,
    }


def rerank_chunks(state: RAGState) -> RAGState:
    """Re-rank retrieved chunks with cross-encoder."""
    query = state.get("rewritten_query") or state["original_query"]
    chunks = state.get("retrieved_chunks", [])

    reranked = rerank(query, chunks)
    return {**state, "reranked_chunks": reranked}


def generate(state: RAGState) -> RAGState:
    """Generate the final grounded answer."""
    query = state["original_query"]
    chunks = state.get("reranked_chunks", [])

    answer = generate_answer(query, chunks)
    return {
        **state,
        "answer": answer,
        "status": "success",
    }


def handle_no_context(state: RAGState) -> RAGState:
    """If context is not relevant, either retry or give a fallback answer."""
    retry = state.get("retry_count", 0) + 1
    max_retries = settings.agent.max_retries

    if retry <= max_retries:
        logger.warning(
            f"🔄 Context not relevant (attempt {retry}/{max_retries}). Retrying..."
        )
        return {**state, "retry_count": retry}
    else:
        logger.warning("❌ Max retries reached. Returning fallback answer.")
        fallback = GeneratedAnswer(
            query=state["original_query"],
            answer=(
                "I couldn't find sufficiently relevant information in the "
                "stock market news database to answer your question. "
                "Please try rephrasing your query or asking about a different topic."
            ),
            is_grounded=False,
        )
        return {
            **state,
            "answer": fallback,
            "status": "success",
        }


# ── Routing functions ────────────────────────────────────────

def route_after_validation(state: RAGState) -> str:
    if state.get("status") == "blocked":
        return "end_blocked"
    return "rewrite_query"


def route_after_relevance(state: RAGState) -> str:
    if state.get("is_relevant_context"):
        return "rerank_chunks"
    return "handle_no_context"


def route_after_no_context(state: RAGState) -> str:
    if state.get("retry_count", 0) <= settings.agent.max_retries:
        if state.get("status") != "success":
            return "rewrite_query"
    return END


# ── Build the graph ──────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """Construct the LangGraph RAG agent."""

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("validate_query", validate_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("rerank_chunks", rerank_chunks)
    graph.add_node("generate", generate)
    graph.add_node("handle_no_context", handle_no_context)

    # Set entry point
    graph.set_entry_point("validate_query")

    # Conditional edges
    graph.add_conditional_edges(
        "validate_query",
        route_after_validation,
        {
            "end_blocked": END,
            "rewrite_query": "rewrite_query",
        },
    )

    # Linear edges
    graph.add_edge("rewrite_query", "route_query")
    graph.add_edge("route_query", "retrieve")
    graph.add_edge("retrieve", "check_relevance")


    # Conditional: relevant context → rerank, else → handle
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "rerank_chunks": "rerank_chunks",
            "handle_no_context": "handle_no_context",
        },
    )

    graph.add_edge("rerank_chunks", "generate")
    graph.add_edge("generate", END)

    # Retry loop
    graph.add_conditional_edges(
        "handle_no_context",
        route_after_no_context,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        },
    )

    return graph.compile()


# Singleton compiled graph
rag_agent = build_rag_graph()


def run_agent(query: str) -> GeneratedAnswer:
    """Execute the full RAG agent pipeline."""
    initial_state: RAGState = {
        "original_query": query,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": None,
        "is_safe": False,
        "is_relevant_topic": False,
        "is_relevant_context": False,
        "context_relevance_score": 0.0,
        "retry_count": 0,
        "error_message": "",
        "status": "processing",
        "intent": "general",
        "source_filter": None,
    }


    logger.info(f"🤖 Running RAG agent for: {query[:80]}...")
    result = rag_agent.invoke(initial_state)

    if result.get("answer"):
        return result["answer"]

    # If blocked, return the error message as the answer
    return GeneratedAnswer(
        query=query,
        answer=result.get("error_message", "An unexpected error occurred."),
        is_grounded=False,
    )
