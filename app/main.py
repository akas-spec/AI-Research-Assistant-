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
    div[data-testid="column"] .stButton > button {
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
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
    st.metric("Local Documents in DB", stats['num_documents'])

    st.divider()

    st.header("🎯 Query Modes")
    st.markdown("""
    <div style='font-size:0.85rem;'>
    <span class='mode-badge analytical'>📊 ANALYSE</span><br>
    Compare & analyse your <b>uploaded</b> documents<br>
    <i>Retrieval from local DB → Gemini</i>
    <br><br>
    <span class='mode-badge research'>🔬 RESEARCH</span><br>
    Search the latest papers on arXiv<br>
    <i>arXiv Search → Download → Chunk → Temp DB → Retrieve → LLM</i>
    <br><br>
    <span class='mode-badge simple'>⚡ SIMPLE</span><br>
    Direct Groq answer — no document lookup
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
            "e.g. 'Compare the two uploaded papers on attention'  •  "
            "'Latest research on RAG systems'  •  "
            "'What is a transformer?'"
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
            help="Compare & analyse your uploaded PDFs — Retrieval from local DB + Gemini",
        )

    with col_research:
        research_btn = st.button(
            "🔬 Research",
            use_container_width=True,
            help=(
                "Full arXiv pipeline: Search → Download PDFs → Chunk → "
                "Temp Vector DB → Retrieve → LLM"
            ),
        )

    with col_simple:
        simple_btn = st.button(
            "⚡ Quick Answer",
            use_container_width=True,
            help="Direct Groq answer — fast, no document retrieval",
        )

    st.caption(
        "💡 **Analyse** uses your uploaded docs.  "
        "**Research** fetches fresh papers from arXiv and builds a temporary index."
    )

    # ── Determine mode ─────────────────────────────────────────────────────────
    if analysis_btn:
        selected_mode = "analysis"
    elif research_btn:
        selected_mode = "research"
    elif simple_btn:
        selected_mode = "simple"
    else:
        selected_mode = None

    # ── Process ────────────────────────────────────────────────────────────────
    if selected_mode and query.strip():

        # Research mode can take a while — give the user live feedback
        spinner_msg = {
            "analysis": "🔬 Retrieving from local DB and analysing with Gemini...",
            "research": (
                "🌐 Searching arXiv, downloading papers, indexing & retrieving... "
                "(this may take ~30–60 s)"
            ),
            "simple":   "⚡ Generating quick answer...",
        }[selected_mode]

        with st.spinner(spinner_msg):
            result = components['router'].route(query, mode=selected_mode)

        st.session_state.history.append({'query': query, 'result': result})

        st.divider()

        # ── Mode badge + model ─────────────────────────────────────────────
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

        # ── Answer ─────────────────────────────────────────────────────────
        st.markdown("### 📝 Answer")
        st.write(result['answer'])

        # ── Sources ────────────────────────────────────────────────────────
        if result.get('sources'):
            is_research = result['type'] == 'research'
            if is_research:
                unique_sources = {}
                for s in result['sources']:
                  arxiv_id = s.get('arxiv_id')
                  if arxiv_id not in unique_sources:
                       unique_sources[arxiv_id] = s

                display_sources = list(unique_sources.values())
            else:
                display_sources = result['sources']

                
            num_chunks = len(result['sources'])
            num_unique = len(display_sources)

            label = (
               f"📚 arXiv Sources — {num_unique} paper(s), {num_chunks} chunks retrieved"
               if is_research
               else f"📚 Local Sources ({num_chunks} chunks)"
           )

            with st.expander(label):
                for i, source in enumerate(display_sources, 1):
                    if is_research:
                        arxiv_id = source.get('arxiv_id', '')
                        title    = source.get('title', source.get('source_file', 'Unknown'))
                        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None

                        st.markdown(
                            f"**{i}. {title}**"
                            + (f" — [arXiv:{arxiv_id}]({arxiv_url})" if arxiv_url else "")
                            + f"  (Page {source.get('page', 'N/A')})"
                        )
                    else:
                        st.markdown(
                            f"**{i}. {source.get('source_file', 'Unknown')}** "
                            f"(Page {source.get('page', 'N/A')})"
                        )

                    if 'relevance_score' in source:
                        st.caption(f"Relevance: {source['relevance_score']:.3f}")
                    st.divider()

        # ── Extracted info (research mode) ─────────────────────────────────
        if result.get('extracted_info') and show_details:
            with st.expander("🔍 Extracted Information (pre-synthesis)"):
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
    st.info(
        "📝 Upload PDFs here to add them to your **local knowledge base**. "
        "Use **📊 Analyse Documents** to query them. "
        "The **🔬 Research** button fetches fresh papers from arXiv instead."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload research papers to add to your local knowledge base",
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
                    docs = components['loader'].load_pdf_pymupdf(save_path)
                    st.success(f"✅ Loaded {len(docs)} pages")

                    st.info("✂️ Chunking text...")
                    chunks      = components['chunker'].chunk_documents(docs)
                    chunk_stats = components['chunker'].get_chunk_stats(chunks)
                    st.success(f"✅ Created {chunk_stats['num_chunks']} chunks")

                    st.info("💾 Adding to ChromaDB...")
                    st.session_state.vector_store.add_documents(chunks, show_progress=False)
                    st.success("✅ Added to local vector database")

                    st.markdown("---")
                    st.markdown("### ✅ Processing Complete!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pages",    len(docs))
                    c2.metric("Chunks",   chunk_stats['num_chunks'])
                    c3.metric("Avg Size", f"{chunk_stats['avg_chunk_size']:.0f}")

                    new_stats = st.session_state.vector_store.get_collection_stats()
                    st.info(f"📊 Total documents in local database: **{new_stats['num_documents']}**")
                    st.session_state.uploaded_files.append(uploaded_file.name)

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")

    st.divider()
    st.header("📊 Current Local Database")
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