"""
utils/helpers.py - Small reusable utility functions
"""

import os
from typing import List, Dict
from datetime import datetime

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to max length with ellipsis
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def get_timestamp() -> str:
    """
    Get current timestamp as string
    
    Returns:
        Timestamp string (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def count_tokens(text: str) -> int:
    """
    Rough token count estimation (1 token ≈ 4 chars)
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4

def extract_paper_title_from_filename(filename: str) -> str:
    """
    Clean up PDF filename to extract paper title
    
    Args:
        filename: PDF filename
    
    Returns:
        Cleaned title
    """
    # Remove .pdf extension
    title = filename.replace('.pdf', '')
    
    # Replace underscores and hyphens with spaces
    title = title.replace('_', ' ').replace('-', ' ')
    
    # Capitalize words
    title = ' '.join(word.capitalize() for word in title.split())
    
    return title

def validate_pdf_file(file_path: str) -> bool:
    """
    Check if file exists and is a PDF
    
    Args:
        file_path: Path to file
    
    Returns:
        True if valid PDF
    """
    return os.path.exists(file_path) and file_path.endswith('.pdf')

def get_file_info(file_path: str) -> Dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
    
    Returns:
        Dict with file metadata
    """
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    
    return {
        'name': os.path.basename(file_path),
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime)
    }

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')
    
    return text

def format_sources(sources: List[Dict]) -> str:
    """
    Format source list as readable string
    
    Args:
        sources: List of source metadata dicts
    
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        filename = source.get('source_file', 'Unknown')
        page = source.get('page', 'N/A')
        formatted.append(f"{i}. {filename} (Page {page})")
    
    return '\n'.join(formatted)


# Example usage
if __name__ == "__main__":
    # Test utilities
    print(format_file_size(1536000))  # Should print "1.5 MB"
    print(truncate_text("This is a very long text that needs truncating", 20))
    print(get_timestamp())
    print(extract_paper_title_from_filename("attention_is_all_you_need.pdf"))
    print(count_tokens("The quick brown fox jumps over the lazy dog"))