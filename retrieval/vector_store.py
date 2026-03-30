"""
retrieval/vector_store.py - ChromaDB vector store management
"""

from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ingestion.embeddings import EmbeddingGenerator
from app import config
import os

class VectorStore:
    """
    Manages ChromaDB vector store for document retrieval
    
    IMPORTANT: Appends new documents instead of rebuilding
    """
    
    def __init__(self, collection_name: str = "research_papers"):
        """
        Initialize or load existing ChromaDB
        """
        self.collection_name = collection_name
        self.persist_directory = config.CHROMA_DB_PATH
        
        os.makedirs(self.persist_directory, exist_ok=True)
        # Initialize embeddings
        self.embedding_generator = EmbeddingGenerator()
        
        # Load or create vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_generator.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ ChromaDB loaded from {self.persist_directory}")
    
    def add_documents(self, documents: List[Document], show_progress: bool = True):
        """
        Add new documents to existing ChromaDB (APPEND, not rebuild)
        
        Args:
            documents: List of Document objects with chunks
            show_progress: Show progress messages
        """
        if not documents:
            print("⚠️  No documents to add")
            return
        batch_size=200
        if show_progress:
            print(f"📝 Adding {len(documents)} chunks to ChromaDB...")
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                # Add to existing collection (appends)
                self.vectorstore.add_documents(batch)
            #persist after adding all batches    
            self.vectorstore.persist()

            if show_progress:
                print(f"✅ Added {len(documents)} chunks successfully")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store
        """
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            'collection_name': self.collection_name,
            'num_documents': count,
            'persist_directory': self.persist_directory,
            'status': "loaded"
        }
    
    def search_similar(self, query: str, k: int = config.RETRIEVAL_TOP_K,score_threshold: float = None,filter_metadata: dict = None) -> List[Document]:
        """
        Unified search function with scoring, and filtering

        Args:
         query: User query
         k: Number of results
         score_threshold: Optional filter for relevance
         filter_metadata: Optional metadata filter

        Returns:
           List[Document] OR List[(Document, score)]
    """
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score <= score_threshold]

            return results
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def delete_collection(self,confirm: bool = False):
        """
        Delete the entire collection (use with caution!)
        """
        if not confirm:
            print("⚠️  Please confirm deletion with 'confirm=True'")
            return
        try:
            self.vectorstore.delete_collection()
            print(f"🗑️  Collection '{self.collection_name}' deleted successfully")
        except Exception as e:
            print(f"❌ Error during collection deletion: {e}")
    
    def check_if_document_exists(self, source_file: str) -> bool:
        """
        Check if a document is already in the database
        
        Args:
            source_file: Filename to check
        
        Returns:
            True if document exists
        """
        # Search for any chunk with this source file
        try:
         results = self.vectorstore.get(
            where={"metadata": {"source_file":{ "$eq": source_file }}},
            limit=1
         )
         return len(results.get("documents", [])) > 0
        except Exception as e:
            print(f"❌ Error checking document existence: {e}")
            return False


# Example usage
if __name__ == "__main__":
    from ingestion.loader import DocumentLoader
    from ingestion.chunking import DocumentChunker
    
    # Initialize
    store = VectorStore()
    loader = DocumentLoader()
    chunker = DocumentChunker()
    
    # Load and process documents
    all_docs = loader.load_all_pdfs()
    
    for filename, docs in all_docs.items():
        # Check if already exists
        if store.check_if_document_exists(filename):
            print(f"⏭️  {filename} already in database, skipping")
            continue
        
        # Chunk and add
        chunks = chunker.chunk_documents(docs)
        store.add_documents(chunks)
    
    # Show stats
    stats = store.get_collection_stats()
    print(f"\n📊 Database Stats:")
    print(f"  Total chunks: {stats['num_documents']}")
    
    # Test search
    query = "What is a transformer?"
    results = store.search_similar(query, k=3)
    print(f"\n🔍 Search results for '{query}':")
    if not results:
     print("⚠️ No relevant results found")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. {doc.page_content[:100]}... (Score: {score:.2f})")