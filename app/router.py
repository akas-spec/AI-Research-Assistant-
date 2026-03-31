"""
app/router.py - Manual query router (no smart/auto classification)

Three explicit modes selected by the user in the UI:
  1. analysis  → Retrieval + Gemini  (compare uploaded documents)
  2. research  → arXiv Search + Retriever + Context Build + LLM
  3. simple    → Direct Groq answer  (fallback when no button is clicked)
"""

from typing import Dict, Any

from llms.groq_llm import GroqLLM
from llms.gemini_llm import GeminiLLM
from llms.prompts import *
from retrieval.retriever import DocumentRetriever
from tools.arxiv_search import search_arxiv
from app import config


class QueryRouter:
    """
    Manual query router — mode is chosen explicitly by the user,
    not inferred automatically.
    """

    def __init__(self, retriever: DocumentRetriever = None):
        self.groq = GroqLLM()
        self.gemini_pro = GeminiLLM("pro")
        self.gemini_advanced = GeminiLLM("advanced")
        self.retriever = retriever or DocumentRetriever()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def route(self, query: str, mode: str = "simple") -> Dict[str, Any]:
        """
        Route the query to the correct handler based on the user-selected mode.

        Args:
            query: The user's question / prompt.
            mode:  One of  "analysis" | "research" | "simple"
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
            # Covers "simple" and any unexpected value
            return self._handle_simple(query)

    # ------------------------------------------------------------------
    # HANDLER 1 – SIMPLE  (default / no button chosen)
    # ------------------------------------------------------------------
    def _handle_simple(self, query: str) -> Dict[str, Any]:
        """
        Direct Groq call — no retrieval, no external search.
        Used when the user submits a query without pressing either button.
        """
        print("🚀 Mode: SIMPLE → Direct Groq")

        prompt = format_prompt(SIMPLE_PROMPT, query=query)
        answer = self.groq.invoke(prompt, temperature=0.3, max_tokens=800)

        return {
            "answer": answer,
            "type": "simple",
            "model": "groq-llama-70b",
            "sources": [],
            "pipeline": ["answer"],
        }

    # ------------------------------------------------------------------
    # HANDLER 2 – ANALYSIS  (button: "📊 Analyse Documents")
    # ------------------------------------------------------------------
    def _handle_analytical(self, query: str) -> Dict[str, Any]:
        """
        Retrieval + Gemini analysis.
        Retrieves chunks from the uploaded documents stored in ChromaDB,
        then uses Gemini for deep comparison / analytical reasoning.
        """
        print("🔬 Mode: ANALYSIS → Retrieval + Gemini")

        # Retrieve broader context for comparison
        context_data = self.retriever.get_context_for_query(
            query,
            k=config.RETRIEVAL_TOP_K + 2,
        )

        # Build prompt
        prompt = format_prompt(
            ANALYTICAL_PROMPT,
            query=query,
            context=context_data["context"],
        )

        # Pick Gemini tier based on complexity keywords
        gemini = self.gemini_pro
        answer = gemini.invoke(prompt)

        return {
            "answer": answer,
            "type": "analytical",
            "model": f"{gemini.model_name} + chromadb",
            "sources": context_data["sources"],
            "num_chunks": context_data["num_chunks"],
            "pipeline": ["retrieve_multi", "analyze"],
        }

    # ------------------------------------------------------------------
    # HANDLER 3 – RESEARCH  (button: "🔬 Research")
    # ------------------------------------------------------------------
    def _handle_research(self, query: str) -> Dict[str, Any]:
        """
        Full research pipeline:
          Query → arXiv Search → Retriever → Context Build → LLM

        Steps:
          1. Search arXiv for relevant papers.
          2. Retrieve related chunks from local ChromaDB.
          3. Merge both into a rich combined context.
          4. Groq extracts structured info (fast & cheap).
          5. Gemini synthesises a detailed answer with sources.
        """
        print("🤖 Mode: RESEARCH → Full Pipeline")

        # Step 1 – arXiv search
        print("  → Searching arXiv...")
        papers_text = search_arxiv(query, max_results=5)

        # Step 2 – Local retrieval
        print("  → Retrieving from local DB...")
        context_data = self.retriever.get_context_for_query(
            query,
            k=config.RETRIEVAL_TOP_K,
        )

        # Step 3 – Build combined context
        combined_context = (
            f"=== arXiv Papers ===\n{papers_text}\n\n"
            f"=== Local Documents ===\n{context_data['context']}"
        )

        # Step 4 – Extract key info (Groq — fast)
        print("  → Extracting key information...")
        extract_prompt = format_prompt(
            EXTRACTION_PROMPT,
            papers_content=combined_context,
        )
        extracted_info = self.groq.invoke(
            extract_prompt,
            temperature=0.2,
            max_tokens=2000,
        )

        print("  → Synthesising final answer...")
        synthesis_prompt = format_prompt(
            RESEARCH_SYNTHESIS_PROMPT,
            query=query,
            extracted_info=extracted_info,
        )
        gemini = self.gemini_advanced
        answer = gemini.invoke(synthesis_prompt)

        return {
            "answer": answer,
            "type": "research",
            "model": f"groq (extract) + {gemini.model_name} (synthesise)",
            "sources": context_data["sources"],
            "extracted_info": extracted_info,
            "num_papers": 5,
            "pipeline": ["arxiv_search", "retrieve", "context_build", "extract", "synthesise"],
        }