from typing import List, Dict, Tuple
from langchain.schema import Document
from retrieval.vector_store import VectorStore
from app import config


class DocumentRetriever:
    """
    Retrieves relevant document chunks for queries
    """

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        filter_metadata: Dict = None
    ) -> List[Document]:
        """
        Retrieve only documents (no scores)
        """
        results = self.vector_store.search_similar(
            query,
            k=k,
            filter_metadata=filter_metadata
        )

        # Extract only documents
        return [doc for doc, _ in results]

    def retrieve_with_scores(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        filter_metadata: Dict = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        """
        return self.vector_store.search_similar(
            query,
            k=k,
            filter_metadata=filter_metadata
        )

    def get_context_for_query(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        include_metadata: bool = True
    ) -> Dict:
        """
        Get formatted context for LLM
        """
        results = self.retrieve_with_scores(query, k=k)

        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}\n")

            if include_metadata:
                sources.append({
                    'source_file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'relevance_score': float(score)
                })

        return {
            'context': "\n".join(context_parts),
            'sources': sources,
            'num_chunks': len(results)
        }

    def retrieve_from_specific_paper(
        self,
        query: str,
        paper_name: str,
        k: int = 3
    ) -> List[Document]:
        """
        Retrieve chunks from a specific paper using metadata filtering
        """

        results = self.retrieve(
            query,
            k=k,
            filter_metadata={"source_file": paper_name}
        )

        return results
    
if __name__ == "__main__":

        retriever = DocumentRetriever()
    
    # Test query
        query = "What is attention mechanism?"
    
    # Basic retrieval
        docs = retriever.retrieve(query, k=3)
        print(f"Retrieved {len(docs)} documents\n")
    
        for i, doc in enumerate(docs, 1):
         print(f"{i}. {doc.page_content[:150]}...")
         print(f"   Source: {doc.metadata.get('source_file', 'Unknown')}\n")
    
    # Get formatted context
        context_data = retriever.get_context_for_query(query, k=3)
    
        print(f"\n📝 Context for LLM ({context_data['num_chunks']} chunks):")
        print(context_data['context'][:300])
    
        print("\n📚 Sources:")
        for src in context_data["sources"]:
          print(src)