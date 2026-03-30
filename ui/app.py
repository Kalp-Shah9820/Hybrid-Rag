"""
Streamlit UI — Interactive chat interface for the RAG Stock Market system.

Run with: streamlit run ui/app.py
"""

import time

import streamlit as st
import httpx

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Stock Market RAG Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #a0a0b0;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    .citation-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .citation-card:hover {
        border-color: #7b2ff7;
        box-shadow: 0 0 15px rgba(123, 47, 247, 0.2);
    }

    .citation-title {
        color: #00d2ff;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .citation-snippet {
        color: #c0c0d0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d2ff;
    }

    .metric-label {
        color: #808090;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .stChatMessage {
        border-radius: 12px !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Session state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 Stock Market RAG Intelligence</h1>
    <p>Production-grade hybrid retrieval with citation grounding</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    # Health check
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        health = resp.json()
        if health.get("db_connected"):
            st.success("🟢 Database Connected")
        else:
            st.warning("🟡 Database Disconnected")

        # Stats
        resp = httpx.get(f"{API_BASE}/stats", timeout=5)
        stats = resp.json()
        st.markdown("---")
        st.markdown("### 📊 Database Stats")
        st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")
        st.caption(f"Embedding: `{stats.get('embedding_model', 'N/A')}`")
        st.caption(f"Re-ranker: `{stats.get('reranker_model', 'N/A')}`")
    except Exception:
        st.error("🔴 API Offline — Start with: `python -m uvicorn api.main:app --reload`")

    st.markdown("---")
    st.markdown("### 📋 Sample Queries")
    samples = [
        "What are the latest trends in the stock market?",
        "How did Tesla stock perform recently?",
        "What impact did the Fed rate decision have?",
        "Which sectors showed strongest growth?",
        "What are analysts predicting for tech stocks?",
    ]
    for sample in samples:
        if st.button(sample, key=f"sample_{hash(sample)}", use_container_width=True):
            st.session_state.pending_query = sample

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #606070; font-size: 0.8rem;'>"
        "Built with LangGraph • ChromaDB • BM25 • FastAPI</p>",

        unsafe_allow_html=True,
    )

# ── Chat display ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citations if present
        if msg.get("citations"):
            with st.expander(f"📚 {len(msg['citations'])} Source(s)", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(
                        f"""<div class="citation-card">
                            <div class="citation-title">[Source {cit['source_index']}] {cit.get('title', '')}</div>
                            <div class="citation-snippet">{cit.get('snippet', '')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Show latency if present
        if msg.get("latency_ms"):
            st.caption(f"⏱️ {msg['latency_ms']:.0f}ms")

# ── Chat input ───────────────────────────────────────────────
query = st.chat_input("Ask about stock market news...")

# Handle sample query clicks
if hasattr(st.session_state, "pending_query"):
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching & analyzing..."):
            try:
                t0 = time.time()
                resp = httpx.post(
                    f"{API_BASE}/ask",
                    json={"query": query},
                    timeout=60,
                )
                elapsed = (time.time() - t0) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer received.")
                    citations = data.get("citations", [])
                    latency = data.get("latency_ms", elapsed)

                    st.markdown(answer)

                    # Show citations
                    if citations:
                        with st.expander(
                            f"📚 {len(citations)} Source(s)", expanded=False
                        ):
                            for cit in citations:
                                st.markdown(
                                    f"""<div class="citation-card">
                                        <div class="citation-title">[Source {cit['source_index']}] {cit.get('title', '')}</div>
                                        <div class="citation-snippet">{cit.get('snippet', '')}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    st.caption(f"⏱️ {latency:.0f}ms")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "latency_ms": latency,
                    })
                    st.session_state.total_queries += 1
                else:
                    err = f"❌ API Error ({resp.status_code}): {resp.text}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                    })
            except httpx.ConnectError:
                err = "❌ Cannot connect to API. Start it with: `python -m uvicorn api.main:app --reload`"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                })
            except Exception as e:
                err = f"❌ Unexpected error: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                })
