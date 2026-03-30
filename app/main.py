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
    .route-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    .simple { background: #e3f2fd; color: #1976d2; }
    .retrieval { background: #f3e5f5; color: #7b1fa2; }
    .analytical { background: #fff3e0; color: #f57c00; }
    .research { background: #fce4ec; color: #c2185b; }
    .upload-success { background: #e8f5e9; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'router' not in st.session_state:
    st.session_state.router = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize components (cached)
@st.cache_resource
def init_components():
    """Initialize all components"""
    print("🔧 Initializing components...")
    
    vector_store = VectorStore()
    retriever = DocumentRetriever(vector_store)
    router = QueryRouter(retriever)
    
    return {
        'vector_store': vector_store,
        'retriever': retriever,
        'router': router,
        'loader': DocumentLoader(),
        'chunker': DocumentChunker()
    }

# Load components
with st.spinner("🔧 Loading AI models..."):
    components = init_components()
    st.session_state.router = components['router']
    st.session_state.vector_store = components['vector_store']

# Header
st.title("🔬 AI Research Assistant")
st.caption("Smart routing with Gemini Pro + Groq • 100% Free")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # API status
    groq_status = "✅" if config.GROQ_API_KEY else "❌"
    gemini_status = "✅" if config.GEMINI_API_KEY else "❌"
    
    st.markdown(f"""
    **API Keys:**
    - Groq: {groq_status}
    - Gemini: {gemini_status}
    """)
    
    # Database stats
    stats = st.session_state.vector_store.get_collection_stats()
    st.metric("Documents in DB", stats['num_documents'])
    
    # Usage stats
    st.header("📈 Usage (Last Minute)")
    groq_calls = len(getattr(st.session_state.router.groq, "calls", []))
    gemini_calls = (
      len(getattr(st.session_state.router.gemini_pro, "calls", [])) +
      len(getattr(st.session_state.router.gemini_advanced, "calls", []))
    )
    
    col1, col2 = st.columns(2)
    col1.metric("Groq", f"{groq_calls}/30")
    col2.metric("Gemini", f"{gemini_calls}/60")
    
    st.divider()
    
    # Query type legend
    st.header("🎯 Query Types")
    st.markdown("""
    <div style='font-size: 0.85rem;'>
    <span class='route-badge simple'>SIMPLE</span><br>
    Direct answer, no docs
    
    <span class='route-badge retrieval'>RETRIEVAL</span><br>
    Search your papers
    
    <span class='route-badge analytical'>ANALYTICAL</span><br>
    Compare & analyze
    
    <span class='route-badge research'>RESEARCH</span><br>
    Full synthesis
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Uploaded files
    st.header("📄 Uploaded Papers")
    pdf_files = [f for f in os.listdir(config.DATA_RAW_PATH) if f.endswith('.pdf')]
    if pdf_files:
        for pdf in pdf_files:
            st.text(f"• {pdf}")
    else:
        st.caption("No papers uploaded yet")

# Main content - Tabs
tab1, tab2 = st.tabs(["💬 Ask Questions", "📤 Upload Papers"])

# ========== TAB 1: QUERY INTERFACE ==========
with tab1:
    st.header("Ask a Research Question")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., 'What is attention mechanism?', 'Compare BERT and GPT', 'Find papers on RAG'..."
    )
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        show_details = st.checkbox("Show routing details", value=True)
    
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Example queries
    with st.expander("💡 Try These Examples"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📘 What is a transformer?"):
                query = "What is a transformer?"
                search_button = True
            
            if st.button("🔍 Find papers on vision transformers"):
                query = "Find papers on vision transformers in my database"
                search_button = True
        
        with col2:
            if st.button("⚖️ Compare BERT and GPT"):
                query = "Compare BERT and GPT architectures"
                search_button = True
            
            if st.button("🔬 State of multimodal learning"):
                query = "What is the current state of research on multimodal learning?"
                search_button = True
    
    # Process query
    if search_button and query:
        with st.spinner("🤖 Processing..."):
            result = st.session_state.router.route(query)
            
            # Add to history
            st.session_state.history.append({
                'query': query,
                'result': result
            })
            
            st.divider()
            
            # Show route badge
            route_type = result['type'].upper()
            route_colors = {
                'SIMPLE': 'simple',
                'RETRIEVAL': 'retrieval',
                'ANALYTICAL': 'analytical',
                'RESEARCH': 'research'
            }
            
            st.markdown(f"""
            <div style='margin-bottom: 1rem;'>
            <span class='route-badge {route_colors[route_type]}'>{route_type}</span>
            <span style='color: #666; margin-left: 1rem; font-size: 0.85rem;'>
            Model: {result['model']}
            </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show pipeline (if enabled)
            if show_details:
                st.caption(f"Pipeline: {' → '.join(result['pipeline'])}")
            
            # Main answer
            st.markdown("### 📝 Answer")
            st.write(result['answer'])
            
            # Show sources (if available)
            if result.get('sources'):
                with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f"**{i}. {source.get('source_file', 'Unknown')}** (Page {source.get('page', 'N/A')})")
                        if 'relevance_score' in source:
                            st.caption(f"Relevance: {source['relevance_score']:.3f}")
                        st.divider()
            
            # Show extracted info (for research queries)
            if result.get('extracted_info') and show_details:
                with st.expander("🔍 Extracted Information"):
                    st.write(result['extracted_info'])
    
    # Clear history
    if clear_btn:
        st.session_state.history = []
        st.rerun()
    
    # Show history
    if st.session_state.history:
        st.divider()
        st.header("📜 Recent Queries")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"{i}. {item['query'][:60]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Type:** `{item['result']['type']}`")
                st.markdown(f"**Answer:** {item['result']['answer'][:300]}...")

# ========== TAB 2: UPLOAD INTERFACE ==========
with tab2:
    st.header("📤 Upload Research Papers")
    
    st.info("📝 **Important:** PDFs are saved to `data/raw/` and processed into ChromaDB")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload research papers to add to your knowledge base"
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
                    # Step 1: Save to data/raw/
                    st.info("💾 Saving PDF to data/raw/...")
                    filename = os.path.basename(uploaded_file.name)
                    save_path = os.path.join(config.DATA_RAW_PATH, filename)
                    
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Saved to {save_path}")
                    
                    # Step 2: Load document
                    st.info("📖 Loading document...")
                    loader = components['loader']
                    docs = loader.load_pdf(uploaded_file.name)
                    st.success(f"✅ Loaded {len(docs)} pages")
                    
                    # Step 3: Chunk document
                    st.info("✂️ Chunking text...")
                    chunker = components['chunker']
                    chunks = chunker.chunk_documents(docs)
                    stats = chunker.get_chunk_stats(chunks)
                    st.success(f"✅ Created {stats['num_chunks']} chunks")
                    
                    # Step 4: Add to ChromaDB
                    st.info("💾 Adding to ChromaDB...")
                    st.session_state.vector_store.add_documents(chunks, show_progress=False)
                    st.success("✅ Added to vector database")
                    
                    # Show summary
                    st.markdown("---")
                    st.markdown("### ✅ Processing Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Pages", len(docs))
                    col2.metric("Chunks", stats['num_chunks'])
                    col3.metric("Avg Size", f"{stats['avg_chunk_size']:.0f}")
                    
                    # Update database stats
                    new_stats = st.session_state.vector_store.get_collection_stats()
                    st.info(f"📊 Total documents in database: **{new_stats['num_documents']}**")
                    
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
    
    # Show current database
    st.divider()
    st.header("📊 Current Database")
    
    db_stats = st.session_state.vector_store.get_collection_stats()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks", db_stats['num_documents'])
    col2.metric("Papers", len([f for f in os.listdir(config.DATA_RAW_PATH) if f.endswith('.pdf')]))

# Footer
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Powered by Groq (Llama 3.1) + Google Gemini Pro • 100% Free Tier")
with col2:
    if st.button("🔄 Reload App"):
        st.rerun()