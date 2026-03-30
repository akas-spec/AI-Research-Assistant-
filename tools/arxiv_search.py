"""
tools/arxiv_search.py - Arxiv paper search utility
"""

import arxiv
from typing import List, Dict
import time

def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arxiv for papers and return formatted summaries
    
    Args:
        query: Search query
        max_results: Maximum number of papers to return
    
    Returns:
        Formatted string with paper information
    """
    time.sleep(3)
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client(page_size=max_results)
        papers = []
        for result in client.results(search):
            papers.append(f"""
                Title: {result.title}
                Authors: {', '.join([a.name for a in result.authors][:3])}{'...' if len(result.authors) > 3 else ''}
                Published: {result.published.strftime('%Y-%m-%d')}
                Summary: {result.summary}
                URL: {result.entry_id}
            """)
        
        if not papers:
            return "No papers found for this query."
        
        return "\n".join(papers)
        
    except Exception as e:
        return f"Error searching arxiv: {e}"

def get_paper_details(paper_id: str) -> Dict:
    """
    Get detailed information about a specific paper
    
    Args:
        paper_id: Arxiv paper ID (e.g., '2301.00001')
    
    Returns:
        Dict with paper details
    """
    try:
        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        results = next(client.results(search))
        if not results:
            return {'error': 'Paper not found'}
        paper=results[0]
        return {
            'title': paper.title,
            'authors': [a.name for a in paper.authors],
            'abstract': paper.summary,
            'published': paper.published,
            'pdf_url': paper.pdf_url,
            'categories': paper.categories
        }
    except Exception as e:
        return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    # Test search
    results = search_arxiv("attention mechanism transformers", max_results=3)
    print(results)