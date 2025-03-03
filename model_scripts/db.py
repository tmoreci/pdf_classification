import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from pdf_utils import summary_extraction
import os
import re
from tqdm import tqdm
from google import genai


class DocumentDatabase:
    """Handles document storage, indexing, and retrieval operations"""

    def __init__(
        self,
        gemini_api,
        db_path="../data/chroma_db",
        collection_name="academic_papers",
        model_name="all-MiniLM-L6-v2",
    ):
        # ! Todo test out different embedding models
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

    def add_document(self, file_path, doc_id):
        """Process and add a document to the database with metadata"""
        # Extract abstract
        llm_extraction = summary_extraction(file_path, self.gemini_client)

        self.collection.add(
            ids=[f"{doc_id}"],
            documents=[llm_extraction.summary],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "title": llm_extraction.title,
                    # "abstract": llm_extraction.abstract,
                }
            ],
        )
        pass

    def index_directory(self, directory_path):
        """Index all PDF documents in a directory"""
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

    def _tokenize(self, text):
        """Tokenize text for BM25 search"""
        text = re.sub(r"[^\w\s]", "", text.lower())
        return text.split()

    def vector_search(self, query, n_results=5):
        """
        Perform a semantic search on the document collection using vector embeddings.

        This function leverages vector embeddings to find documents that are semantically
        similar to the input query. It queries the collection of documents stored in the
        chroma database and retrieves the top N results that best match the semantic content of
        the query. The search is based on the vector representation of the documents and
        the query, allowing for a more nuanced understanding of the content beyond simple
        keyword matching.

        Parameters:
        - query (str): The natural language query for which the search is to be performed.
        - n_results (int, optional): The number of top matching documents to retrieve.
          Defaults to 5.

        Returns:
        - results (dict): A dictionary containing the search results, which includes the
          documents, their metadata, and associated IDs that are most relevant to the query.
        """
        results = self.collection.query(
            query_texts=[query], n_results=n_results
        )
        return results

    def keyword_search(self, query, field="abstract", n_results=5):
        """Keyword search on metadata using BM25"""
        # Get all documents with their metadata
        all_docs = self.collection.get()

        if not all_docs["metadatas"]:
            return {"documents": [], "metadatas": [], "ids": []}

        # Extract the specified metadata field
        texts = []
        valid_indices = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if field in metadata and metadata[field]:
                texts.append(metadata[field])
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
        }

    def hybrid_search(
        self,
        query,
        field="abstract",
        n_results=5,
        semantic_weight=0.8,
        keyword_weight=0.2,
    ):
        """
        Enhanced hybrid search combining vector and keyword search with weighted rankings.
        Based on Anthropic contextual document embeddings implementation

        Parameters:
        - query (str): The search query
        - field (str): Field to search in for keyword search
        - n_results (int): Number of results to return
        - semantic_weight (float): Weight for semantic search results (0-1)
        - keyword_weight (float): Weight for keyword search results (0-1)

        Returns:
        - dict: Combined search results with ranking information
        """
        # Use a larger initial recall size for better reranking
        initial_recall = max(n_results * 5, 50)

        # Get results from both methods
        vector_results = self.vector_search(query, initial_recall)
        keyword_results = self.keyword_search(query, field, initial_recall)

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
