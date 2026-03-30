"""
ingestion/loader.py - PDF document loading with PyPDFLoader
"""

import os
import fitz  # PyMuPDF for PDF metadata
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from app import config

class DocumentLoader:
    """
    Loads PDF documents from data/raw/ directory
    """
    
    def __init__(self):
        
        self.raw_path = config.DATA_RAW_PATH
    

    def load_pdf_pymupdf(self,file_path):
       docs = []
       pdf = fitz.open(file_path)
    
       for i, page in enumerate(pdf):
        text = page.get_text()
        
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": i
                }
            )
        )
    
       pdf.close()
       return docs   
    def load_all_pdfs(self):
       all_docs = {}
       folder_path=self.raw_path
       for file in os.listdir(folder_path):
         if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
              return (f"File {file_path} does not exist.")
            docs = self.load_pdf_pymupdf(file_path)
            all_docs[file] = docs
            
       return all_docs
    
    def get_document_info(self, pdf_path: str) -> Dict:
        """
        Get metadata about a PDF without loading full content
        """
        file_path = os.path.join(self.raw_path, pdf_path)
        
        if not os.path.exists(file_path):
            return None
        
        # Quick load to get page count
        loader = fitz.open(file_path)
        num_pages = loader.page_count
        loader.close()
        
        return {
            'filename': pdf_path,
            'num_pages': num_pages,
            'file_size': os.path.getsize(file_path),
            'path': file_path
        }


# Example usage
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Load all PDFs
    all_docs = loader.load_all_pdfs()
    
    for filename, docs in all_docs.items():
        print(f"\n{filename}:")
        print(f"  Pages: {len(docs)}")
        print(f"  First page preview: {docs[0].page_content[:200]}...")