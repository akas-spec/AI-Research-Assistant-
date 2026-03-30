"""
ingestion/embeddings.py - Document embedding using HuggingFace
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from app import config

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using HuggingFace models
    100% FREE - runs locally
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading embedding model: {model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if GPU available)
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        
        print("Embedding model loaded")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query string
        
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        """
        # Test with a sample text
        sample = self.embed_query("test")
        return len(sample)


# Example usage
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Test embedding
    test_texts = [
        "Transformers are neural network architectures.",
        "Attention mechanisms are key components.",
        "BERT and GPT are popular models."
    ]
    
    # Embed documents
    doc_embeddings = generator.embed_documents(test_texts)
    print(f"\nGenerated {len(doc_embeddings)} embeddings")
    print(f"Embedding dimension: {len(doc_embeddings[0])}")
    
    # Embed query
    query = "What is a transformer?"
    query_embedding = generator.embed_query(query)
    print(f"\nQuery embedding dimension: {len(query_embedding)}")