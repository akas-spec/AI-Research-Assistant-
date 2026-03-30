"""
app/router.py - Smart query router with dynamic routing
"""

from enum import Enum
from typing import Dict, Any

from click import prompt
from llms.groq_llm import GroqLLM
from llms.gemini_llm import GeminiLLM
from llms.prompts import *
from retrieval.retriever import DocumentRetriever
from tools.arxiv_search import search_arxiv
from app import config

class QueryType(Enum):
    SIMPLE = "simple"
    RETRIEVAL = "retrieval"
    ANALYTICAL = "analytical"
    RESEARCH = "research"

class QueryRouter:
    """
    Intelligent query router with dynamic pipeline selection
    """
    
    def __init__(self, retriever: DocumentRetriever = None):
        self.groq = GroqLLM()
        self.gemini_pro = GeminiLLM("pro") 
        self.gemini_advanced = GeminiLLM("advanced")
        self.retriever = retriever or DocumentRetriever()
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type using fast Groq model
        """
        prompt = format_prompt(CLASSIFICATION_PROMPT, query=query)
        
        try:
            result = self.groq.classify(prompt)
            category = result.split('|')[0].strip().upper()
            
            print(f"📊 Query classified as: {category}")
            if category not in QueryType.__members__:
              return QueryType.SIMPLE
            
            if "SIMPLE" in category:
                return QueryType.SIMPLE
            elif "RETRIEVAL" in category:
                return QueryType.RETRIEVAL
            elif "ANALYTICAL" in category:
                return QueryType.ANALYTICAL
            else:
                return QueryType.RESEARCH
        except Exception as e:
            print(f"⚠️  Classification error: {e}. Defaulting to SIMPLE")
            return QueryType.SIMPLE
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Main routing function - classifies and routes to appropriate handler
        """
        # Step 1: Classify
        query_type = self.classify_query(query)
        
        # Step 2: Route
        if query_type == QueryType.SIMPLE:
            return self._handle_simple(query)
        elif query_type == QueryType.RETRIEVAL:
            return self._handle_retrieval(query)
        elif query_type == QueryType.ANALYTICAL:
            return self._handle_analytical(query)
        else:
            return self._handle_research(query)
    
    # ========== HANDLER 1: SIMPLE ==========
    def _handle_simple(self, query: str) -> Dict[str, Any]:
        """
        Direct LLM call - no retrieval
        Route: Groq (fast)
        """
        print("🚀 Route: SIMPLE → Direct Groq")
        
        prompt = format_prompt(SIMPLE_PROMPT, query=query)
        answer = self.groq.invoke(prompt, temperature=0.3, max_tokens=800)
        
        return {
            'answer': answer,
            'type': 'simple',
            'model': 'groq-llama-70b',
            'sources': [],
            'pipeline': ['classify', 'answer']
        }
    
    # ========== HANDLER 2: RETRIEVAL ==========
    def _handle_retrieval(self, query: str) -> Dict[str, Any]:
        """
        RAG pipeline: Retrieve → Answer
        Route: Groq + ChromaDB
        """
        print("📚 Route: RETRIEVAL → RAG Pipeline (Groq)")
        
        # Retrieve context
        context_data = self.retriever.get_context_for_query(
          query, 
          k=config.RETRIEVAL_TOP_K
        )
        
        # Generate answer
        prompt = format_prompt(
            RETRIEVAL_PROMPT,
            query=query,
            context=context_data['context']
        )
        answer = self.groq.invoke(prompt, temperature=0.3, max_tokens=1200)
        
        return {
            'answer': answer,
            'type': 'retrieval',
            'model': 'groq-llama-70b + chromadb',
            'sources': context_data['sources'],
            'num_chunks': context_data['num_chunks'],
            'pipeline': ['classify', 'retrieve', 'answer']
        }
    
    # ========== HANDLER 3: ANALYTICAL ==========
    def _handle_analytical(self, query: str) -> Dict[str, Any]:
        """
        Multi-doc analysis: Retrieve → Analyze
        Route: Gemini (better reasoning)
        """
        print("🔬 Route: ANALYTICAL → Gemini Analysis")
        
        # Retrieve more context for analysis
        context_data = self.retriever.get_context_for_query(
           query, 
           k=config.RETRIEVAL_TOP_K + 2
        )
        
        # Deep analysis with Gemini
        prompt = format_prompt(
            ANALYTICAL_PROMPT,
            query=query,
            context=context_data['context']
        )
        query_lower = query.lower()   
        COMPLEX_KEYWORDS = [
             "detailed architecture",
             "deep research",
             "deep analysis",
             "detailed analysis",
             "compare",
        ]
        if any(word in query_lower for word in COMPLEX_KEYWORDS):
          gemini = self.gemini_advanced  
        else:
          gemini = self.gemini_pro       

        answer = gemini.invoke(prompt)
        return {
            'answer': answer,
            'type': 'analytical',
            'model': f'{gemini.model_name} + chromadb',
            'sources': context_data['sources'],
            'num_chunks': context_data['num_chunks'],
            'pipeline': ['classify', 'retrieve_multi', 'analyze']
        }
    
    # ========== HANDLER 4: RESEARCH ==========
    def _handle_research(self, query: str) -> Dict[str, Any]:
        """
        Full pipeline: Search → Extract → Synthesize
        Route: Groq (extraction) + Gemini (synthesis)
        """
        print("🤖 Route: RESEARCH → Full Pipeline")
        
        # Step 1: Search papers
        print("  → Searching papers...")
        papers_text = search_arxiv(query, max_results=5)
        
        # Step 2: Extract key info (Groq - fast)
        print("  → Extracting information...")
        extract_prompt = format_prompt(
            EXTRACTION_PROMPT,
            papers_content=papers_text
        )
        extracted_info = self.groq.invoke(
            extract_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Step 3: Synthesize (Gemini - reasoning)
        print("  → Synthesizing research...")
        synthesis_prompt = format_prompt(
            RESEARCH_SYNTHESIS_PROMPT,
            query=query,
            extracted_info=extracted_info
        )
        query_lower = query.lower()   
        COMPLEX_KEYWORDS = [
             "detailed architecture",
             "deep research",
             "deep analysis",
             "detailed analysis",
             "compare",
        ]
        if any(word in query_lower for word in COMPLEX_KEYWORDS):
          gemini = self.gemini_advanced  
        else:
          gemini = self.gemini_pro       

        answer = gemini.invoke(synthesis_prompt)
        
        return {
            'answer': answer,
            'type': 'research',
            'model': f'groq (extract) + {gemini.model_name}(synthesize)',
            'sources': [],
            'extracted_info': extracted_info,
            'num_papers': 5,
            'pipeline': ['classify', 'search', 'extract', 'synthesize']
        }


# Example usage
if __name__ == "__main__":
    router = QueryRouter()
    
    # Test queries
    queries = [
        "What is a transformer?",
        "Find information about attention mechanisms in my papers",
        "Compare BERT and GPT architectures",
        "What is the current state of research on multimodal learning?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        result = router.route(query)
        print(f"\nType: {result['type']}")
        print(f"Model: {result['model']}")
        print(f"Pipeline: {' → '.join(result['pipeline'])}")
        print(f"\nAnswer: {result['answer'][:200]}...")