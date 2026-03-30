"""
ingestion/chunking.py - Text splitting strategies
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app import config

class DocumentChunker:
    """
    Splits documents into chunks for embedding
    """
    
    def __init__(
        self, 
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""] 
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects from loader
        
        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split raw text into chunks
        
        Args:
            text: Raw text string
            metadata: Optional metadata dict to attach
        
        Returns:
            List of Document chunks
        """
        chunks = self.text_splitter.split_text(text)
        
        # Convert to Document objects
        docs = []
        for i, chunk in enumerate(chunks):
            meta = metadata.copy() if metadata else {}
            meta['chunk_id'] = i
            docs.append(Document(page_content=chunk, metadata=meta))
        
        return docs
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks
        """
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'num_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_lengths) / len(chunks) if chunks else 0,
            'min_chunk_size': min(chunk_lengths) if chunks else 0,
            'max_chunk_size': max(chunk_lengths) if chunks else 0,
            'total_chars': sum(chunk_lengths)
        }


# Example usage
if __name__ == "__main__":
    from loader import DocumentLoader
    
    # Load documents
    loader = DocumentLoader()
    all_docs = loader.load_all_pdfs()
    
    # Chunk them
    chunker = DocumentChunker()
    all_chunks=[]
    for filename, docs in all_docs.items():
        chunks = chunker.chunk_documents(docs)
        stats = chunker.get_chunk_stats(chunks)
        all_chunks.extend(chunks)
        print(f"\n{filename}:")
        print(f"  Original pages: {len(docs)}")
        print(f"  Chunks created: {stats['num_chunks']}")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")