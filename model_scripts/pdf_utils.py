from tqdm import tqdm
import pymupdf
import os
import time
from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from prompts import summary_prompt, abstract_prompt
from base import Extraction, Summary
from typing import Dict, List, Any, Optional, Union


def full_text_parse(file_path: str) -> str:
    """
    Parse the full text content of a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        String containing the full text content of the PDF
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    document_parts = []  # Initialize a list to collect parts of the document
    for i, page in enumerate(documents):
        document_parts.append(f"Page_{i}\n{page.page_content}\n")
    document = "".join(document_parts)
    return document


def add_paper_chunks(
    pdf_path: str,
    paper_id: str,
    collection: Any,
    metadata: Optional[Dict[str, Any]] = None,
    add_context: bool = False,
) -> int:
    """
    Split a PDF into chunks and add them to a ChromaDB collection.

    Args:
        pdf_path: Path to the PDF file
        paper_id: Unique identifier for the paper
        collection: ChromaDB collection to add chunks to
        metadata: Additional metadata to store with each chunk
        add_context: Whether to add context to each chunk using Gemini

    Returns:
        Number of chunks added to the collection
    """
    # Load and split the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Add to collection with metadata
    for i, chunk in enumerate(chunks):
        if add_context:
            chunk.page_content = gemini_add_context(
                pdf_path, chunk.page_content
            )
        collection.add(
            ids=[f"{paper_id}_chunk_{i}"],
            documents=[chunk.page_content],
            metadatas=[
                {
                    "paper_id": paper_id,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_id": i,
                    **(metadata or {}),
                }
            ],
        )

    return len(chunks)


def add_papers(db_path: str, db_name: str, pdfs_path: str, gemini: Any) -> None:
    """
    Add multiple papers from a directory to a ChromaDB collection.

    Args:
        db_path: Path to the ChromaDB database
        db_name: Name of the collection
        pdfs_path: Path to the directory containing PDF files
        gemini: Initialized Gemini client
    """
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # Initialize ChromaDB (persistent)
    client = chromadb.PersistentClient(path=db_path)

    # Create or get collection
    collection = client.get_or_create_collection(
        name=db_name, embedding_function=embedding_function
    )
    files = os.listdir(pdfs_path)
    for i, file in tqdm(enumerate(files)):
        full_file_path = os.path.join(pdfs_path, file)

        response = summary_extraction(full_file_path, gemini)
        print(response)
        success = False
        while not success:
            try:
                collection.add(
                    ids=[f"doc_{i}"],
                    documents=[response.summary],
                    metadatas=[
                        {
                            "title": response.title,
                        }
                    ],
                )
                success = True
            except Exception as e:
                print(f"Error adding chunk {i}: {e}. Retrying in 30 seconds...")
                time.sleep(30)


def gemini_add_context(file: str, chunk: str, client: Any) -> str:
    """
    Add contextual information to a document chunk using Gemini.

    Args:
        file: Path to the PDF file
        chunk: Text chunk to add context to
        client: Initialized Gemini client

    Returns:
        Chunk with added context
    """
    chunk_addition = f"""
<chunk> 
{chunk}
</chunk> 
The provided chunk is a small subsection of the full PDF document you have been provided. Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
    # Convert the file path string to a Path object
    file_path = Path(file)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type="application/pdf",
            ),
            chunk_addition,
        ],
    )
    output = f"{chunk}\n\n{response.text}"
    print(response.text)
    return output


def summary_extraction(file_path: str, client: Any) -> Summary:
    """
    Extract title and summary from scientific article PDFs using Gemini API.

    Args:
        file_path: Path to the PDF file
        client: Initialized Gemini API client

    Returns:
        Parsed Summary object with title and summary
    """
    # Open the document with PyMuPDF
    file_path = Path(file_path)

    # Configure Gemini model
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": Summary,
    }

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type="application/pdf",
            ),
            summary_prompt,
        ],
        config=generation_config,
    )
    return response.parsed


def abstract_extraction(file_path: str, client: Any) -> Extraction:
    """
    Extract title and abstract from scientific article PDFs using Gemini API.

    Args:
        file_path: Path to the PDF file
        client: Initialized Gemini API client

    Returns:
        Parsed Extraction object with title and abstract
    """

    # Open the document with PyMuPDF
    with pymupdf.open(file_path) as document:
        # Create a new PDF writer
        writer = pymupdf.open()

        # Copy the first 5 pages (or fewer if the document is shorter)
        for page_number in range(min(5, len(document))):
            writer.insert_pdf(
                document, from_page=page_number, to_page=page_number
            )

        # Save the first 5 pages to a bytes buffer
        pdf_bytes = writer.tobytes()

    # Configure Gemini model
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": Extraction,
    }

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            ),
            abstract_prompt,
        ],
        config=generation_config,
    )
    return response.parsed


def query_papers(
    query_text: str, collection: Any, n_results: int = 5
) -> Dict[str, Any]:
    """
    Query a collection of papers with a search query.

    Args:
        query_text: The search query
        collection: ChromaDB collection to search
        n_results: Number of results to return

    Returns:
        Search results from the collection
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results
