"""
app/router.py - Manual query router (no smart/auto classification)

Three explicit modes selected by the user in the UI:
  1. analysis  → Retrieval + Gemini  (compare uploaded local documents)
  2. research  → Full arXiv pipeline:
                   User Query
                       ↓
                   arXiv Search
                       ↓
                   Download / Extract Paper Content
                       ↓
                   Chunking
                       ↓
                   Store in Temporary Session Vector DB
                       ↓
                   Retriever (Top-K from THESE papers only)
                       ↓
                   Context
                       ↓
                   LLM (detailed answer + sources)
  3. simple    → Direct Groq answer (fallback when no button is clicked)
"""

import os
import uuid
import tempfile
import requests
from typing import Dict, Any, List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from llms.groq_llm import GroqLLM
from llms.gemini_llm import GeminiLLM
from llms.prompts import *
from retrieval.retriever import DocumentRetriever
from ingestion.chunking import DocumentChunker
from tools.arxiv_search import search_arxiv
from app import config


# ---------------------------------------------------------------------------
# Arxiv paper downloader — fetches full PDF text via PyMuPDF
# ---------------------------------------------------------------------------
def _download_arxiv_papers(query: str, max_results: int = 5):
    """
    Optimized arXiv downloader:
    - Limits to top 2 papers
    - Limits pages per PDF
    - Uses caching
    - Parallel downloads
    """

    import arxiv
    import fitz
    import requests
    import tempfile
    import os
    from concurrent.futures import ThreadPoolExecutor
    from langchain.schema import Document

    global arxiv_cache, paper_cache
    if "arxiv_cache" not in globals():
        arxiv_cache = {}
    if "paper_cache" not in globals():
        paper_cache = {}

    MAX_PAPERS = 2        
    MAX_PAGES  = 12      

    client = arxiv.Client()

    # ─────────────────────────────────────────────
    # Step 1: Cache arXiv results
    # ─────────────────────────────────────────────
    if query in arxiv_cache:
        papers = arxiv_cache[query]
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = list(client.results(search))
        arxiv_cache[query] = papers

    # ─────────────────────────────────────────────
    # Step 2: Select TOP papers
    # (simple ranking: longer abstract = more info)
    # ─────────────────────────────────────────────
    papers.sort(key=lambda p: len(p.summary or ""), reverse=True)
    top_papers = papers[:MAX_PAPERS]

    # ─────────────────────────────────────────────
    # Step 3: Process each paper (parallel)
    # ─────────────────────────────────────────────
    def process_paper(result):
        docs = []

        paper_title = result.title
        paper_id = result.entry_id.split("/")[-1]

        print(f"  📄 Processing: {paper_title[:60]}...")

        # ── CACHE CHECK ──────────────────────────
        if paper_id in paper_cache:
            return paper_cache[paper_id]

        try:
            pdf_url = result.pdf_url
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                pdf = fitz.open(tmp_path)

                for page_num, page in enumerate(pdf):
                    if page_num >= MAX_PAGES:
                        break   # 🔥 LIMIT PAGES

                    text = page.get_text().strip()
                    if not text:
                        continue

                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source_file": f"{paper_id}.pdf",
                            "title": paper_title,
                            "arxiv_id": paper_id,
                            "page": page_num,
                            "source": "arxiv",
                        }
                    ))

                pdf.close()

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            print(f"    ⚠️ PDF failed ({e}), using abstract.")

            abstract = result.summary or ""
            if abstract:
                docs.append(Document(
                    page_content=f"Title: {paper_title}\n\nAbstract:\n{abstract}",
                    metadata={
                        "source_file": f"{paper_id}_abstract.txt",
                        "title": paper_title,
                        "arxiv_id": paper_id,
                        "page": 0,
                        "source": "arxiv_abstract",
                    }
                ))

        # ── SAVE TO CACHE ────────────────────────
        paper_cache[paper_id] = docs
        return docs

    # ─────────────────────────────────────────────
    # Parallel execution
    # ─────────────────────────────────────────────
    all_docs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_paper, top_papers)

    for docs in results:
        all_docs.extend(docs)

    return all_docs


# ---------------------------------------------------------------------------
# Temporary (session-scoped) vector store
# ---------------------------------------------------------------------------

def _build_temp_vectorstore(documents: List[Document]) -> Chroma:
    """
    Chunk the documents, embed them, and store in an in-memory
    (or temp-dir) Chroma collection that is unique to this request.
    No overlap with the persistent local DB.
    """
    # Chunk
    chunker = DocumentChunker()
    chunks  = chunker.chunk_documents(documents)
    print(f"  ✂️  Created {len(chunks)} chunks from {len(documents)} pages")

    # Embed — reuse the same model config as the rest of the project
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Unique collection name so concurrent sessions don't collide
    collection_name = f"arxiv_session_{uuid.uuid4().hex[:8]}"

    # Use a temp directory — nothing is persisted after the call
    tmp_dir = tempfile.mkdtemp(prefix="arxiv_chroma_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=tmp_dir,
    )

    print(f"  💾 Temp vector DB ready ({collection_name}) — {len(chunks)} chunks")
    return vectorstore


def _retrieve_from_temp_store(
    vectorstore: Chroma,
    query: str,
    k: int,
) -> Dict[str, Any]:
    """
    Run similarity search on the temporary vectorstore and return
    the same context dict format that DocumentRetriever produces
    (so the rest of router stays consistent).
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    context_parts = []
    sources       = []

    for i, (doc, score) in enumerate(results, 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content}\n")
        sources.append({
            "source_file":     doc.metadata.get("source_file", "Unknown"),
            "title":           doc.metadata.get("title", "Unknown"),
            "arxiv_id":        doc.metadata.get("arxiv_id", ""),
            "page":            doc.metadata.get("page", "N/A"),
            "relevance_score": float(score),
        })

    return {
        "context":    "\n".join(context_parts),
        "sources":    sources,
        "num_chunks": len(results),
    }


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Manual query router — mode is chosen explicitly by the user,
    not inferred automatically.
    """

    def __init__(self, retriever: DocumentRetriever = None):
        self.groq            = GroqLLM()
        self.gemini_pro      = GeminiLLM("pro")
        self.gemini_advanced = GeminiLLM("advanced")
        self.retriever       = retriever or DocumentRetriever()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def route(self, query: str, mode: str = "simple") -> Dict[str, Any]:
        """
        Route the query to the correct handler based on the user-selected mode.

        Args:
            query : The user's question / prompt.
            mode  : One of "analysis" | "research" | "simple"
                    Passed directly from the UI button the user clicked.

        Returns:
            Standard result dict consumed by main.py.
        """
        mode = mode.lower().strip()

        if mode == "analysis":
            return self._handle_analytical(query)
        elif mode == "research":
            return self._handle_research(query)
        else:
            return self._handle_simple(query)

    # ------------------------------------------------------------------
    # HANDLER 1 – SIMPLE  (default / no button chosen)
    # ------------------------------------------------------------------
    def _handle_simple(self, query: str) -> Dict[str, Any]:
        """
        Direct Groq call — no retrieval, no external search.
        """
        print("🚀 Mode: SIMPLE → Direct Groq")

        prompt = format_prompt(SIMPLE_PROMPT, query=query)
        answer = self.groq.invoke(prompt, temperature=0.3, max_tokens=800)

        return {
            "answer":   answer,
            "type":     "simple",
            "model":    "groq-llama-70b",
            "sources":  [],
            "pipeline": ["answer"],
        }

    # ------------------------------------------------------------------
    # HANDLER 2 – ANALYSIS  (button: "📊 Analyse Documents")
    # ------------------------------------------------------------------
    def _handle_analytical(self, query: str) -> Dict[str, Any]:
        """
        Retrieval from LOCAL ChromaDB + Gemini analysis.
        Used to compare / analyse documents the user has already uploaded.
        """
        print("🔬 Mode: ANALYSIS → Local Retrieval + Gemini")

        context_data = self.retriever.get_context_for_query(
            query,
            k=config.RETRIEVAL_TOP_K + 2,
        )

        prompt = format_prompt(
            ANALYTICAL_PROMPT,
            query=query,
            context=context_data["context"],
        )

        gemini = self.gemini_pro
        answer = gemini.invoke(prompt)

        return {
            "answer":     answer,
            "type":       "analytical",
            "model":      f"{gemini.model_name} + chromadb",
            "sources":    context_data["sources"],
            "num_chunks": context_data["num_chunks"],
            "pipeline":   ["retrieve_local", "analyze"],
        }

    # ------------------------------------------------------------------
    # HANDLER 3 – RESEARCH  (button: "🔬 Research")
    #
    #   User Query
    #       ↓
    #   arXiv Search
    #       ↓
    #   Download / Extract Paper Content  (PDF → pages)
    #       ↓
    #   Chunking
    #       ↓
    #   Store in Temporary Session Vector DB
    #       ↓
    #   Retriever — Top-K from THESE papers only
    #       ↓
    #   Context
    #       ↓
    #   LLM (Groq extract → Gemini synthesise)
    # ------------------------------------------------------------------
    def _handle_research(self, query: str) -> Dict[str, Any]:
        """
        Full arXiv research pipeline.
        Fetched papers are chunked and stored in a *temporary* vector DB
        that is created fresh for each request — completely isolated from
        the user's persistent local DB.
        """
        print("🤖 Mode: RESEARCH → Full arXiv Pipeline")

        # ── Step 1 : Search arXiv & download paper content ──────────────────
        print("  → Searching arXiv and downloading papers...")
        arxiv_docs = _download_arxiv_papers(query, max_results=5)

        if not arxiv_docs:
            return {
                "answer":   "⚠️ No papers could be retrieved from arXiv for this query. "
                            "Please try a different or more specific query.",
                "type":     "research",
                "model":    "n/a",
                "sources":  [],
                "pipeline": ["arxiv_search", "failed"],
            }

        # ── Step 2 : Chunk + store in temp vector DB ─────────────────────────
        print("  → Chunking and indexing in temporary vector DB...")
        temp_vectorstore = _build_temp_vectorstore(arxiv_docs)

        # ── Step 3 : Retrieve top-K chunks from THESE papers only ────────────
        print("  → Retrieving top-K relevant chunks from arXiv papers...")
        context_data = _retrieve_from_temp_store(
            temp_vectorstore,
            query,
            k=config.RETRIEVAL_TOP_K,
        )

        # ── Step 4 : Extract structured info (Groq — fast & cheap) ──────────
        print("  → Extracting key information...")
        prompt = format_prompt(
           RESEARCH_SYNTHESIS_PROMPT,
           query=query,
           extracted_info=context_data["context"]
        )
        gemini= self.gemini_pro
        answer = gemini.invoke(prompt)
        # ── Cleanup: drop the temp collection ────────────────────────────────
        try:
            temp_vectorstore.delete_collection()
        except Exception:
            pass   # non-critical

        return {
            "answer":         answer,
            "type":           "research",
            "model":          f"groq (extract) + {gemini.model_name} (synthesise)",
            "sources":        context_data["sources"],
            "num_papers":     len({s["arxiv_id"] for s in context_data["sources"]}),
            "num_chunks":     context_data["num_chunks"],
            "pipeline":       [
                "arxiv_search",
                "download_pdfs",
                "chunk",
                "temp_vector_db",
                "retrieve_top_k",
                "extract",
                "synthesise",
            ],
        }