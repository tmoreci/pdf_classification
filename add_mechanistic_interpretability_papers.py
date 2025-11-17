#!/usr/bin/env python3
"""
Script to add mechanistic interpretability LLM papers from arXiv to the database.

This script:
1. Searches arXiv for mechanistic interpretability papers
2. Downloads the PDFs
3. Adds them to the document database
"""

import os
import sys
from dotenv import load_dotenv


from model_scripts.arxiv_scraper import ArxivScraper, add_arxiv_papers_to_database
from model_scripts.db import DocumentDatabase
from model_scripts.db import DocumentDatabase

# Load environment variables
load_dotenv()


def main():
    """Main function to add mechanistic interpretability papers to the database."""
    
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it as an environment variable.")
        return 1
    
    DB_PATH = "data/chroma_db"
    COLLECTION_NAME = "academic_papers"
    DOWNLOAD_DIR = "data/arxiv_papers"
    MAX_RESULTS = 50  # Adjust as needed
    DAYS_BACK = None  # Set to a number (e.g., 30) to only get papers from last N days, or None for all
    
    # Ensure directories exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)
    
    print("=" * 80)
    print("Adding Mechanistic Interpretability Papers to Database")
    print("=" * 80)
    print(f"Database path: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print(f"Max results: {MAX_RESULTS}")
    if DAYS_BACK:
        print(f"Days back: {DAYS_BACK}")
    print("=" * 80)
    print()
    
    # Initialize scraper
    print("Initializing arXiv scraper...")
    scraper = ArxivScraper(max_results=MAX_RESULTS)
    
    # Initialize database
    print("Initializing database...")
    database = DocumentDatabase(
        gemini_api=GEMINI_API_KEY,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Add papers to database
    results = add_arxiv_papers_to_database(
        scraper=scraper,
        database=database,
        gemini_api_key=GEMINI_API_KEY,
        download_dir=DOWNLOAD_DIR,
        search_mechanistic_interpretability=True,
        max_results=MAX_RESULTS,
        days_back=DAYS_BACK
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total papers found: {results['total_papers']}")
    print(f"PDFs downloaded: {results['downloaded']}")
    print(f"Successfully added to database: {results['added_to_db']}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

