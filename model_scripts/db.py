import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from preprocess_pdfs import summary_extraction
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
        """Semantic search using vector embeddings"""
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

    def hybrid_search(self, query, field="abstract", n_results=5):
        """Combined semantic and keyword search"""
        vector_results = self.vector_search(query, n_results)
        keyword_results = self.keyword_search(query, field, n_results)

        # Combine and deduplicate results
        combined_docs = []
        combined_metadata = []
        seen_ids = set()

        # Helper function to process results from either search
        def process_results(results):
            for i, doc_list in enumerate(results["documents"]):
                for j, doc in enumerate(doc_list):
                    doc_id = results["ids"][i][j]
                    if doc_id not in seen_ids:
                        combined_docs.append(doc)
                        combined_metadata.append(results["metadatas"][i][j])
                        seen_ids.add(doc_id)

        # Process both result sets
        process_results(vector_results)
        process_results(keyword_results)

        # Limit to n_results
        combined_docs = combined_docs[:n_results]
        combined_metadata = combined_metadata[:n_results]

        # Format results
        return {
            "documents": [combined_docs],
            "metadatas": [combined_metadata],
            "ids": [list(seen_ids)[:n_results]],
        }
