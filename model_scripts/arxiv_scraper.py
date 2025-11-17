"""
arXiv Paper Scraper

This module provides functionality to scrape papers from arXiv using the arXiv API.
It supports searching by query, category, date range, and other criteria.
"""

import arxiv
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import os
import requests
from tqdm import tqdm


@dataclass
class ArxivPaper:
    """Data class to represent an arXiv paper"""
    entry_id: str
    title: str
    authors: List[str]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    arxiv_id: str
    primary_category: str
    doi: Optional[str] = None
    comment: Optional[str] = None


class ArxivScraper:
    """Scraper for fetching papers from arXiv"""
    
    def __init__(self, max_results: int = 100, sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate):
        """
        Initialize the arXiv scraper.
        
        Args:
            max_results: Maximum number of results to return per query
            sort_by: How to sort results (SubmittedDate, LastUpdatedDate, Relevance)
        """
        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = arxiv.SortOrder.Descending
    
    def search(
        self,
        query: Optional[str] = None,
        id_list: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        sort_by: Optional[arxiv.SortCriterion] = None,
        sort_order: Optional[arxiv.SortOrder] = None
    ) -> List[ArxivPaper]:
        """
        Search arXiv for papers matching the given criteria.
        
        Args:
            query: Search query string (e.g., "machine learning", "cat:cs.AI")
            id_list: List of arXiv IDs to fetch directly (e.g., ["1234.5678", "cs/1234567"])
            max_results: Override default max_results
            sort_by: Override default sort_by
            sort_order: Sort order (Ascending or Descending)
            
        Returns:
            List of ArxivPaper objects
        """
        # Use provided parameters or defaults
        max_results = max_results or self.max_results
        sort_by = sort_by or self.sort_by
        sort_order = sort_order or self.sort_order
        
        # Build search query
        search_query = query if query else None
        
        # Create search object
        search = arxiv.Search(
            query=search_query,
            id_list=id_list,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Fetch results
        papers = []
        for result in search.results():
            paper = self._result_to_paper(result)
            papers.append(paper)
            # Be respectful to arXiv API - small delay between requests
            time.sleep(0.1)
        
        return papers
    
    def search_by_category(
        self,
        category: str,
        max_results: Optional[int] = None,
        days_back: Optional[int] = None
    ) -> List[ArxivPaper]:
        """
        Search for papers in a specific arXiv category.
        
        Args:
            category: arXiv category (e.g., "cs.AI", "cs.LG", "math.OC")
            max_results: Maximum number of results
            days_back: Only return papers from the last N days
            
        Returns:
            List of ArxivPaper objects
        """
        query = f"cat:{category}"
        
        if days_back:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000+TO+{end_date.strftime('%Y%m%d')}235959]"
            query = f"{query} AND {date_query}"
        
        return self.search(query=query, max_results=max_results)
    
    def search_recent(
        self,
        query: Optional[str] = None,
        days_back: int = 7,
        max_results: Optional[int] = None
    ) -> List[ArxivPaper]:
        """
        Search for recent papers (within the last N days).
        
        Args:
            query: Optional search query to filter results
            days_back: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of ArxivPaper objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000+TO+{end_date.strftime('%Y%m%d')}235959]"
        
        if query:
            full_query = f"{query} AND {date_query}"
        else:
            full_query = date_query
        
        return self.search(query=full_query, max_results=max_results)
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., "2301.12345" or "cs/1234567")
            
        Returns:
            ArxivPaper object or None if not found
        """
        results = self.search(id_list=[arxiv_id], max_results=1)
        return results[0] if results else None
    
    def _result_to_paper(self, result: arxiv.Result) -> ArxivPaper:
        """Convert an arxiv.Result to an ArxivPaper dataclass"""
        return ArxivPaper(
            entry_id=result.entry_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            summary=result.summary,
            published=result.published,
            updated=result.updated,
            categories=result.categories,
            pdf_url=result.pdf_url,
            arxiv_id=result.entry_id.split('/')[-1],
            primary_category=result.primary_category,
            doi=result.doi if hasattr(result, 'doi') else None,
            comment=result.comment if hasattr(result, 'comment') else None
        )
    
    def papers_to_dict(self, papers: List[ArxivPaper]) -> List[Dict[str, Any]]:
        """
        Convert a list of ArxivPaper objects to dictionaries.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            List of dictionaries containing paper information
        """
        return [
            {
                "entry_id": paper.entry_id,
                "title": paper.title,
                "authors": paper.authors,
                "summary": paper.summary,
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat(),
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "arxiv_id": paper.arxiv_id,
                "primary_category": paper.primary_category,
                "doi": paper.doi,
                "comment": paper.comment
            }
            for paper in papers
        ]
    
    def download_pdf(self, paper: ArxivPaper, download_dir: str) -> str:
        """
        Download a PDF for an arXiv paper.
        
        Args:
            paper: ArxivPaper object
            download_dir: Directory to save the PDF
            
        Returns:
            Path to the downloaded PDF file
            
        Raises:
            requests.RequestException: If the download fails
            IOError: If the file cannot be written
        """
        os.makedirs(download_dir, exist_ok=True)
        
        # Create a safe filename from the paper title and arxiv_id
        safe_title = "".join(c for c in paper.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:100]  # Limit length
        filename = f"{paper.arxiv_id}_{safe_title}.pdf"
        filename = filename.replace(' ', '_')
        file_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            return file_path
        
        # Download the PDF
        response = requests.get(paper.pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        time.sleep(0.5)  # Be respectful to arXiv servers
        return file_path
    
    def download_papers(self, papers: List[ArxivPaper], download_dir: str) -> List[str]:
        """
        Download PDFs for multiple papers.
        
        Args:
            papers: List of ArxivPaper objects
            download_dir: Directory to save the PDFs
            
        Returns:
            List of paths to downloaded PDF files
            
        Raises:
            requests.RequestException: If any download fails
            IOError: If any file cannot be written
        """
        downloaded_paths = []
        
        for paper in tqdm(papers, desc="Downloading PDFs"):
            pdf_path = self.download_pdf(paper, download_dir)
            downloaded_paths.append(pdf_path)
        
        return downloaded_paths
    
    def search_mechanistic_interpretability(
        self,
        max_results: Optional[int] = None,
        days_back: Optional[int] = None
    ) -> List[ArxivPaper]:
        """
        Search for mechanistic interpretability papers related to LLMs.
        
        This searches for papers on mechanistic interpretability, circuit analysis,
        and related topics in large language models.
        
        Args:
            max_results: Maximum number of results
            days_back: Only return papers from the last N days
            
        Returns:
            List of ArxivPaper objects
        """
        # Build query for mechanistic interpretability papers
        # Focus on LLM interpretability, circuit analysis, and mechanistic understanding
        query_terms = [
            "mechanistic interpretability",
            "circuit analysis",
            "neural circuit",
            "activation patching",
            "causal tracing",
            "transformer interpretability",
            "attention mechanism interpretability"
        ]
        
        # Combine terms with OR
        query = " OR ".join([f'"{term}"' for term in query_terms])
        
        # Add LLM/large language model context
        query = f"({query}) AND (large language model OR LLM OR transformer OR GPT)"
        
        # Add date filter if specified
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000+TO+{end_date.strftime('%Y%m%d')}235959]"
            query = f"{query} AND {date_query}"
        
        return self.search(query=query, max_results=max_results)


def add_arxiv_papers_to_database(
    scraper: ArxivScraper,
    database,
    gemini_api_key: str,
    download_dir: str = "data/arxiv_papers",
    papers: Optional[List[ArxivPaper]] = None,
    search_mechanistic_interpretability: bool = False,
    max_results: Optional[int] = None,
    days_back: Optional[int] = None
) -> Dict[str, Any]:
    """
    Add arXiv papers to the database by downloading PDFs and indexing them.
    
    This function can either:
    1. Use a provided list of papers
    2. Search for mechanistic interpretability papers automatically
    
    Args:
        scraper: ArxivScraper instance
        database: DocumentDatabase instance
        gemini_api_key: Gemini API key for summary extraction
        download_dir: Directory to save downloaded PDFs
        papers: Optional list of ArxivPaper objects to add
        search_mechanistic_interpretability: If True, search for mechanistic interpretability papers
        max_results: Maximum number of papers to add (if searching)
        days_back: Only search papers from the last N days (if searching)
        
    Returns:
        Dictionary with statistics about the operation
    """
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Get papers to process
    if papers is None:
        if search_mechanistic_interpretability:
            print("Searching for mechanistic interpretability papers...")
            papers = scraper.search_mechanistic_interpretability(
                max_results=max_results,
                days_back=days_back
            )
            print(f"Found {len(papers)} papers")
        else:
            raise ValueError("Either provide papers or set search_mechanistic_interpretability=True")
    
    # Download PDFs
    print(f"\nDownloading {len(papers)} PDFs...")
    paper_to_path = {}
    for paper in tqdm(papers, desc="Downloading PDFs"):
        pdf_path = scraper.download_pdf(paper, download_dir)
        paper_to_path[paper.arxiv_id] = (paper, pdf_path)
    
    print(f"Successfully downloaded {len(paper_to_path)} PDFs")
    
    # Add papers to database
    print(f"\nAdding papers to database...")
    added_count = 0
    
    for arxiv_id, (paper, pdf_path) in paper_to_path.items():
        # Use arxiv_id as the document ID
        doc_id = f"arxiv_{arxiv_id}"
        database.add_document(pdf_path, doc_id)
        added_count += 1
        print(f"Added: {paper.title[:60]}...")
    
    return {
        "total_papers": len(papers),
        "downloaded": len(paper_to_path),
        "added_to_db": added_count
    }


# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = ArxivScraper(max_results=10)
    
    # Example 1: Search by query
    print("Searching for 'machine learning' papers...")
    papers = scraper.search(query="machine learning", max_results=5)
    print(f"Found {len(papers)} papers\n")
    
    for paper in papers:
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"Published: {paper.published}")
        print(f"PDF URL: {paper.pdf_url}")
        print(f"Categories: {', '.join(paper.categories)}")
        print("-" * 80)
    
    # Example 2: Search by category
    print("\n\nSearching for recent cs.AI papers...")
    ai_papers = scraper.search_by_category("cs.AI", max_results=5, days_back=7)
    print(f"Found {len(ai_papers)} recent AI papers\n")
    
    # Example 3: Search recent papers
    print("\n\nSearching for recent papers in the last 3 days...")
    recent_papers = scraper.search_recent(days_back=3, max_results=5)
    print(f"Found {len(recent_papers)} papers from the last 3 days\n")

