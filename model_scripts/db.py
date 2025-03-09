import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from pdf_utils import summary_extraction
import os
import re
from tqdm import tqdm
from google import genai
from typing import Dict, List, Any, Optional, Tuple, Union


class DocumentDatabase:
    """Handles document storage, indexing, and retrieval operations"""

    def __init__(
        self,
        gemini_api: str,
        db_path: str = "../data/chroma_db",
        collection_name: str = "academic_papers",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the document database with embedding functions and ChromaDB client.

        Args:
            gemini_api: API key for Gemini model access
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to use
            model_name: Name of the embedding model to use
        """
        # Initialize embedding function
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        )

        # Initialize ChromaDB
        self.db_client = chromadb.PersistentClient(path=db_path)

        # Create or get collection
        self.collection = self.db_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        self.gemini_client = genai.Client(api_key=gemini_api)

    def add_document(self, file_path: str, doc_id: str) -> None:
        """
        Process and add a document to the database with metadata.

        Args:
            file_path: Path to the PDF file to process
            doc_id: Unique identifier for the document
        """
        # Extract abstract
        llm_extraction = summary_extraction(file_path, self.gemini_client)

        self.collection.add(
            ids=[f"{doc_id}"],
            documents=[llm_extraction.summary],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "title": llm_extraction.title,
                }
            ],
        )

    def index_directory(self, directory_path: str) -> int:
        """
        Index all PDF documents in a directory.

        Args:
            directory_path: Path to the directory containing PDF files

        Returns:
            Number of documents successfully indexed
        """
        files = os.listdir(directory_path)
        indexed_count = 0

        for i, file in tqdm(
            enumerate(files), total=len(files), desc="Indexing documents"
        ):
            if file.lower().endswith(".pdf"):
                full_file_path = os.path.join(directory_path, file)
                doc_id = f"doc_{i}"

                self.add_document(full_file_path, doc_id)

                indexed_count += 1

        return indexed_count

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search.

        Args:
            text: The text to tokenize

        Returns:
            List of tokens
        """
        text = re.sub(r"[^\w\s]", "", text.lower())
        return text.split()

    def vector_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Perform a semantic search on the document collection using vector embeddings.

        Args:
            query: The natural language query for which the search is to be performed
            n_results: The number of top matching documents to retrieve (default: 5)

        Returns:
            Dictionary containing the search results with documents, metadata, and IDs
        """
        results = self.collection.query(
            query_texts=[query], n_results=n_results
        )
        return results

    def keyword_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Perform keyword search on document content using BM25 algorithm.

        This method applies BM25 ranking directly to the document content rather than
        metadata fields, which provides better full-text search capabilities.

        Args:
            query: The search query text
            n_results: Number of results to return

        Returns:
            Dictionary with search results containing documents, metadatas, and ids
        """
        # Get all documents
        all_docs = self.collection.get()

        if not all_docs["documents"]:
            return {"documents": [], "metadatas": [], "ids": []}

        # Extract the document texts and filter out empty documents
        texts = []
        valid_indices = []

        for i, doc in enumerate(all_docs["documents"]):
            if doc and doc.strip():  # Ensure document has content
                texts.append(doc)
                valid_indices.append(i)

        # Tokenize the texts for BM25
        tokenized_corpus = [self._tokenize(text) for text in texts]

        # Apply BM25 ranking
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(self._tokenize(query))

        # Get top N results
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]
        original_indices = [valid_indices[i] for i in top_indices]

        # Format results similar to vector_search
        return {
            "documents": [[all_docs["documents"][i]] for i in original_indices],
            "metadatas": [[all_docs["metadatas"][i]] for i in original_indices],
            "ids": [[all_docs["ids"][i]] for i in original_indices],
            "scores": [
                [scores[i]] for i in top_indices
            ],  # Include scores in the response
        }

    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        semantic_weight: float = 0.8,
        keyword_weight: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Enhanced hybrid search combining vector and keyword search with weighted rankings.
        Based on Anthropic contextual document embeddings implementation.

        Args:
            query: The search query
            n_results: Number of results to return
            semantic_weight: Weight for semantic search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)

        Returns:
            Dictionary with combined search results and ranking information
        """
        # Use a larger initial recall size for better reranking
        initial_recall = max(n_results * 5, 50)

        # Get results from both methods
        vector_results = self.vector_search(query, initial_recall)
        keyword_results = self.keyword_search(query, initial_recall)

        # Track and score all retrieved document IDs
        doc_scores = {}

        # Process vector search results
        for i, doc_list in enumerate(vector_results["documents"]):
            for j, doc in enumerate(doc_list):
                doc_id = vector_results["ids"][i][j]
                # Use reciprocal rank for scoring
                score = semantic_weight * (1.0 / (j + 1))
                doc_scores[doc_id] = {
                    "score": score,
                    "document": doc,
                    "metadata": vector_results["metadatas"][i][j],
                    "from_vector": True,
                    "from_keyword": False,
                }

        # Process keyword search results
        for i, doc_list in enumerate(keyword_results["documents"]):
            for j, doc in enumerate(doc_list):
                doc_id = keyword_results["ids"][i][j]
                # Use reciprocal rank for scoring
                keyword_score = keyword_weight * (1.0 / (j + 1))

                if doc_id in doc_scores:
                    # Document already found in vector search
                    doc_scores[doc_id]["score"] += keyword_score
                    doc_scores[doc_id]["from_keyword"] = True
                else:
                    # New document from keyword search
                    doc_scores[doc_id] = {
                        "score": keyword_score,
                        "document": doc,
                        "metadata": keyword_results["metadatas"][i][j],
                        "from_vector": False,
                        "from_keyword": True,
                    }

        # Sort by score in descending order
        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )[:n_results]

        # Format results
        documents = []
        metadatas = []
        ids = []
        origins = {"vector_only": 0, "keyword_only": 0, "both": 0}

        for doc_id, info in sorted_results:
            documents.append(info["document"])
            metadatas.append(info["metadata"])
            ids.append(doc_id)

            # Track result origins for analytics
            if info["from_vector"] and info["from_keyword"]:
                origins["both"] += 1
            elif info["from_vector"]:
                origins["vector_only"] += 1
            else:
                origins["keyword_only"] += 1

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "ids": [ids],
            "origins": origins,
            "scores": [info["score"] for _, info in sorted_results],
        }
