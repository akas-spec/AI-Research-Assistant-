"""
app/main.py - Main Streamlit application
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.router import QueryRouter
from ingestion.loader import DocumentLoader
from ingestion.chunking import DocumentChunker
from ingestion.embeddings import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from retrieval.retriever import DocumentRetriever
import config

# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
    .mode-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    .simple    { background: #e3f2fd; color: #1976d2; }
    .analytical{ background: #fff3e0; color: #f57c00; }
    .research  { background: #fce4ec; color: #c2185b; }
    .upload-success { background: #e8f5e9; padding: 1rem; border-radius: 0.5rem; }
    /* Make the two main action buttons stand out */
    div[data-testid="column"] .stButton > button {
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []


# ── Cached component initialisation ───────────────────────────────────────────
@st.cache_resource
def init_components():
    print("🔧 Initialising components...")
    vector_store = VectorStore()
    retriever    = DocumentRetriever(vector_store)
    router       = QueryRouter(retriever)
    return {
        'vector_store': vector_store,
        'retriever':    retriever,
        'router':       router,
        'loader':       DocumentLoader(),
        'chunker':      DocumentChunker(),
    }


with st.spinner("🔧 Loading AI models..."):
    components = init_components()
    st.session_state.vector_store = components['vector_store']


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔬 AI Research Assistant")
st.caption("Powered by Groq (Llama 3.1) + Google Gemini Pro • 100 % Free")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Status")

    groq_status   = "✅" if config.GROQ_API_KEY   else "❌"
    gemini_status = "✅" if config.GEMINI_API_KEY else "❌"

    st.markdown(f"""
    **API Keys:**
    - Groq: {groq_status}
    - Gemini: {gemini_status}
    """)

    stats = st.session_state.vector_store.get_collection_stats()
    st.metric("Documents in DB", stats['num_documents'])

    st.divider()

    st.header("🎯 Query Modes")
    st.markdown("""
    <div style='font-size:0.85rem;'>
    <span class='mode-badge analytical'>ANALYSIS</span><br>
    Compare & analyse your uploaded documents (Retrieval + Gemini)
    <br><br>
    <span class='mode-badge research'>RESEARCH</span><br>
    Full pipeline: arXiv Search → Retrieve → Build Context → LLM
    <br><br>
    <span class='mode-badge simple'>SIMPLE</span><br>
    Direct Groq answer — no button needed, just hit Enter
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.header("📄 Uploaded Papers")
    pdf_files = [f for f in os.listdir(config.DATA_RAW_PATH) if f.endswith('.pdf')]
    if pdf_files:
        for pdf in pdf_files:
            st.text(f"• {pdf}")
    else:
        st.caption("No papers uploaded yet")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask Questions", "📤 Upload Papers"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – QUERY INTERFACE
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Ask a Research Question")

    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder=(
            "e.g. 'What is attention mechanism?'  •  "
            "'Compare the two uploaded papers'  •  "
            "'Latest research on RAG systems'"
        ),
    )

    show_details = st.checkbox("Show pipeline details", value=True)

    # ── Three-column button row ────────────────────────────────────────────────
    st.markdown("**Choose how to process your question:**")
    col_analysis, col_research, col_simple = st.columns(3)

    with col_analysis:
        analysis_btn = st.button(
            "📊 Analyse Documents",
            use_container_width=True,
            help="Compare & analyse your uploaded PDFs using Retrieval + Gemini",
        )

    with col_research:
        research_btn = st.button(
            "🔬 Research",
            use_container_width=True,
            help="Full pipeline: arXiv Search → Retrieve → Context → LLM",
        )

    with col_simple:
        simple_btn = st.button(
            "⚡ Quick Answer",
            use_container_width=True,
            help="Direct Groq answer — fast, no document retrieval",
        )

    st.caption(
        "💡 **Tip:** If you just press Enter / submit without a button, "
        "a quick Groq answer is returned automatically."
    )

    # ── Determine mode ─────────────────────────────────────────────────────────
    if analysis_btn:
        selected_mode = "analysis"
    elif research_btn:
        selected_mode = "research"
    elif simple_btn:
        selected_mode = "simple"
    else:
        selected_mode = None   # nothing pressed yet

    # ── Process ────────────────────────────────────────────────────────────────
    if selected_mode and query.strip():
        st.session_state.stop_generation = False

    # 🔴 Add STOP button HERE
        stop_placeholder = st.empty()

        stop_clicked = stop_placeholder.button("⛔ Stop Response", use_container_width=True)

        with st.spinner("🤖 Processing..."):
            result = components['router'].route(query, mode=selected_mode)

            # Save to history
            st.session_state.history.append({'query': query, 'result': result})

            st.divider()

            # Mode badge
            badge_class = {
                'simple':     'simple',
                'analytical': 'analytical',
                'research':   'research',
            }.get(result['type'], 'simple')

            st.markdown(f"""
            <div style='margin-bottom:1rem;'>
            <span class='mode-badge {badge_class}'>{result['type'].upper()}</span>
            <span style='color:#666; margin-left:1rem; font-size:0.85rem;'>
                Model: {result['model']}
            </span>
            </div>
            """, unsafe_allow_html=True)

            if show_details:
                st.caption("Pipeline: " + " → ".join(result['pipeline']))

            # Answer
            st.markdown("### 📝 Answer")

            placeholder = st.empty()
            response_text = ""

# Choose correct LLM
            if result['type'] == "simple":
              llm = components['router'].groq
            elif result['type'] == "analytical":
              llm = components['router'].gemini_pro
            else:
              llm = components['router'].gemini_advanced

# ⚡ STREAMING LOOP
            for chunk in llm.stream(query):

              if stop_clicked or st.session_state.stop_generation:
                st.warning("⛔ Response stopped by user")
                break

              response_text += chunk
              placeholder.markdown(response_text)

            # Sources
            if result.get('sources'):
                with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(
                            f"**{i}. {source.get('source_file', 'Unknown')}** "
                            f"(Page {source.get('page', 'N/A')})"
                        )
                        if 'relevance_score' in source:
                            st.caption(f"Relevance: {source['relevance_score']:.3f}")
                        st.divider()

            # Extracted info (research mode)
            if result.get('extracted_info') and show_details:
                with st.expander("🔍 Extracted Information"):
                    st.write(result['extracted_info'])

    elif selected_mode and not query.strip():
        st.warning("⚠️ Please enter a question before clicking a button.")

    # ── Clear button ───────────────────────────────────────────────────────────
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    # ── History ────────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        st.header("📜 Recent Queries")
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"{i}. {item['query'][:60]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Mode:** `{item['result']['type']}`")
                st.markdown(f"**Answer:** {item['result']['answer'][:300]}...")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – UPLOAD INTERFACE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📤 Upload Research Papers")
    st.info("📝 PDFs are saved to `data/raw/` and processed into ChromaDB")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload research papers to add to your knowledge base",
    )

    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ File selected: **{uploaded_file.name}**")
            st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
        with col2:
            process_btn = st.button("📥 Process", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing document..."):
                try:
                    st.info("💾 Saving PDF to data/raw/...")
                    filename  = os.path.basename(uploaded_file.name)
                    save_path = os.path.join(config.DATA_RAW_PATH, filename)
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Saved to {save_path}")

                    st.info("📖 Loading document...")
                    docs = components['loader'].load_pdf(uploaded_file.name)
                    st.success(f"✅ Loaded {len(docs)} pages")

                    st.info("✂️ Chunking text...")
                    chunks = components['chunker'].chunk_documents(docs)
                    chunk_stats = components['chunker'].get_chunk_stats(chunks)
                    st.success(f"✅ Created {chunk_stats['num_chunks']} chunks")

                    st.info("💾 Adding to ChromaDB...")
                    st.session_state.vector_store.add_documents(chunks, show_progress=False)
                    st.success("✅ Added to vector database")

                    st.markdown("---")
                    st.markdown("### ✅ Processing Complete!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pages",    len(docs))
                    c2.metric("Chunks",   chunk_stats['num_chunks'])
                    c3.metric("Avg Size", f"{chunk_stats['avg_chunk_size']:.0f}")

                    new_stats = st.session_state.vector_store.get_collection_stats()
                    st.info(f"📊 Total documents in database: **{new_stats['num_documents']}**")
                    st.session_state.uploaded_files.append(uploaded_file.name)

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")

    st.divider()
    st.header("📊 Current Database")
    db_stats = st.session_state.vector_store.get_collection_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total Chunks", db_stats['num_documents'])
    c2.metric("Papers", len([f for f in os.listdir(config.DATA_RAW_PATH) if f.endswith('.pdf')]))


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Powered by Groq (Llama 3.1) + Google Gemini Pro • 100 % Free Tier")
with col2:
    if st.button("🔄 Reload App"):
        st.rerun()