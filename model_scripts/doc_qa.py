import os
import re
import chromadb
from chromadb.utils import embedding_functions
import cohere
import yaml
from preprocess_pdfs import add_paper, gemini_extraction
from tqdm import tqdm
from prompts import rag_preamble
from rank_bm25 import BM25Okapi


class DocumentDatabase:
    """Handles document storage, indexing, and retrieval operations"""

    def __init__(
        self,
        db_path="../data/chroma_db",
        collection_name="academic_papers",
        model_name="all-MiniLM-L6-v2",
    ):
        # Initialize embedding function
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_document(self, file_path, doc_id):
        """Process and add a document to the database with metadata"""
        # Extract abstract
        llm_extraction = gemini_extraction(file_path)

        self.collection.add(
            ids=[f"{doc_id}"],
            documents=[llm_extraction.abstract],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "title": llm_extraction.title,
                    "abstract": llm_extraction.abstract,
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
        print(len(texts))
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


class LLMQueryEngine:
    """Handles LLM interactions for Q&A over retrieved documents"""

    def __init__(self, api_key, config_path="llm_config.yaml"):
        # Initialize LLM client
        self.llm = cohere.ClientV2(api_key)

        # Load configuration
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.model = self.config.get("model", "command-r-08-2024")
        self.temperature = self.config.get("temperature", 0.1)

    def format_documents_for_llm(self, search_results):
        """Convert search results to format expected by LLM"""
        documents = []
        for i, doc in enumerate(search_results["documents"][0]):
            documents.append({"id": str(i), "data": {"text": doc}})
        return documents

    def generate_answer(self, question, documents, temperature=None):
        """Generate answer using retrieved documents"""
        if temperature is None:
            temperature = self.temperature

        # Generate response
        response = self.llm.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": rag_preamble,
                },
                {"role": "user", "content": question},
            ],
            documents=documents,
            temperature=temperature,
            seed=42,
        )

        return response.message.content[0].text


class DocumentQA:
    """Main class that coordinates document retrieval and question answering"""

    def __init__(
        self,
        cohere_api_key,
        db_path="../data/chroma_db",
        collection_name="academic_papers",
    ):
        self.db = DocumentDatabase(db_path, collection_name)
        self.llm = LLMQueryEngine(cohere_api_key)

    def index_documents(self, directory_path):
        """Index all documents in a directory"""
        return self.db.index_directory(directory_path)

    def answer_question(
        self,
        question,
        search_type="semantic",
        metadata_field="abstract",
        n_results=5,
        temperature=None,
    ):
        """Answer a question using RAG with the specified search strategy"""
        # Retrieve relevant documents
        if search_type == "semantic":
            results = self.db.vector_search(question, n_results)
        elif search_type == "keyword":
            results = self.db.keyword_search(
                question, metadata_field, n_results
            )
        elif search_type == "hybrid":
            results = self.db.hybrid_search(question, metadata_field, n_results)
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        # Format documents for the LLM
        formatted_docs = self.llm.format_documents_for_llm(results)

        # Generate answer
        answer = self.llm.generate_answer(question, formatted_docs, temperature)

        # Return both the answer and sources
        return {
            "answer": answer,
            "sources": [
                {"text": doc, "metadata": results["metadatas"][0][i]}
                for i, doc in enumerate(results["documents"][0])
            ],
        }


# Example usage
if __name__ == "__main__":
    cohere_api_key = ""

    # Initialize QA system
    qa = DocumentQA(cohere_api_key)

    # Index documents (uncomment to run indexing)
    # qa.index_documents("../data")

    # Answer questions
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() in ["quit", "exit", "q"]:
            break

        # Allow selecting search type
        print("\nSearch options:")
        print("1. Semantic search (vector similarity)")
        print("2. Keyword search on abstracts (BM25)")
        print("3. Hybrid search (combines both methods)")

        search_option = input("Select search type (1-3, default=1): ").strip()

        if search_option == "2":
            search_type = "keyword"
        elif search_option == "3":
            search_type = "hybrid"
        else:
            search_type = "semantic"

        result = qa.answer_question(question, search_type=search_type)

        print("\n" + "=" * 80)
        print(f"Using {search_type} search")
        print("Answer:")
        print(result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["sources"]):
            print(f"\n--- Source {i+1} ---")
            metadata = source["metadata"]
            print(
                f"Paper ID: {metadata.get('paper_id')}, Page: {metadata.get('page')}"
            )
            if "abstract" in metadata:
                print(
                    f"From document with abstract: {metadata['abstract'][:100]}..."
                )
            print(f"Text snippet: {source['text'][:150]}...")
        print("=" * 80)
