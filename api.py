"""
FastAPI application for PDF Q&A system.
Provides RESTful API endpoints for document management, search, and question answering.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_scripts.db import DocumentDatabase
from model_scripts.doc_qa import GeminiLLM, CohereLLM
from model_scripts.pdf_utils import summary_extraction, abstract_extraction, full_text_parse
from model_scripts.topic_clustering import TopicClustering
from api_schemas import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    SearchRequest,
    HybridSearchRequest,
    SearchResult,
    SearchResponse,
    QARequest,
    QAResponse,
    CitedDocument,
    PDFExtractionResponse,
    BasicTopicRequest,
    ZeroShotTopicRequest,
    TopicModelingResponse,
    TopicInfo,
    DocumentTopicInfo,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

# Configuration
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "academic_papers"
MODEL_NAME = "all-MiniLM-L6-v2"
TEMP_DIR = "data/temp"
API_VERSION = "1.0.0"

# Global database instance
db: Optional[DocumentDatabase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global db
    # Startup
    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    db = DocumentDatabase(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        model_name=MODEL_NAME
    )
    yield
    # Shutdown
    if db:
        # ChromaDB handles cleanup automatically
        pass


# Initialize FastAPI app
app = FastAPI(
    title="PDF Q&A API",
    description="API for managing academic papers, searching, and question answering with LLM",
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health and System Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    database_accessible = db is not None
    return HealthResponse(
        status="healthy" if database_accessible else "unhealthy",
        database_accessible=database_accessible,
        version=API_VERSION
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get database statistics."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        collection_info = db.collection.get()
        total_docs = len(collection_info['ids']) if collection_info['ids'] else 0

        return StatsResponse(
            total_documents=total_docs,
            collection_name=COLLECTION_NAME,
            embedding_model=MODEL_NAME,
            database_path=DB_PATH
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    doc_id: Optional[str] = Form(None, description="Custom document ID (optional)")
):
    """
    Upload and index a PDF document.

    The document will be processed to extract title and summary, then indexed
    in the vector database for searching.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Generate doc_id if not provided
        if not doc_id:
            doc_id = Path(file.filename).stem

        # Add document to database
        db.add_document(temp_file_path, doc_id)

        # Get the document info
        result = db.collection.get(ids=[doc_id])
        if result['ids']:
            title = result['metadatas'][0].get('title', doc_id)
            return DocumentUploadResponse(
                doc_id=doc_id,
                title=title,
                message="Document uploaded and indexed successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Document indexed but could not retrieve info")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/documents/list", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """
    List all indexed documents in the database.

    Returns document IDs, titles, and summaries.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        result = db.collection.get()

        documents = []
        if result['ids']:
            for i, doc_id in enumerate(result['ids']):
                documents.append(DocumentInfo(
                    doc_id=doc_id,
                    title=result['metadatas'][i].get('title', doc_id),
                    summary=result['documents'][i] if result['documents'] else ""
                ))

        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/documents/{doc_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(doc_id: str):
    """
    Get details of a specific document.

    Returns the document's title and summary.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        result = db.collection.get(ids=[doc_id])

        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        return DocumentInfo(
            doc_id=doc_id,
            title=result['metadatas'][0].get('title', doc_id),
            summary=result['documents'][0] if result['documents'] else ""
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse, tags=["Documents"])
async def delete_document(doc_id: str):
    """
    Delete a document from the database.

    Removes the document and its embeddings from the vector database.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        # Check if document exists
        result = db.collection.get(ids=[doc_id])
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Delete the document
        db.collection.delete(ids=[doc_id])

        return DocumentDeleteResponse(
            doc_id=doc_id,
            message="Document deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/search/vector", response_model=SearchResponse, tags=["Search"])
async def vector_search(request: SearchRequest):
    """
    Perform semantic vector search on indexed documents.

    Uses embeddings to find semantically similar documents.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        results = db.vector_search(request.query, n_results=request.n_results)

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            search_results.append(SearchResult(
                doc_id=doc_id,
                title=results['metadatas'][0][i].get('title', doc_id),
                content=results['documents'][0][i],
                score=1 - results['distances'][0][i] if results['distances'] else None
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing vector search: {str(e)}")


@app.post("/search/keyword", response_model=SearchResponse, tags=["Search"])
async def keyword_search(request: SearchRequest):
    """
    Perform BM25 keyword search on indexed documents.

    Uses traditional keyword matching to find relevant documents.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        results = db.keyword_search(request.query, n_results=request.n_results)

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            search_results.append(SearchResult(
                doc_id=doc_id,
                title=results['metadatas'][0][i].get('title', doc_id),
                content=results['documents'][0][i],
                score=None  # BM25 scores not returned in current implementation
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing keyword search: {str(e)}")


@app.post("/search/hybrid", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search combining vector and keyword search.

    Combines semantic understanding with keyword matching for better results.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        results = db.hybrid_search(
            request.query,
            n_results=request.n_results,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight
        )

        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            search_results.append(SearchResult(
                doc_id=doc_id,
                title=results['metadatas'][0][i].get('title', doc_id),
                content=results['documents'][0][i],
                score=None  # Hybrid scores not returned in current implementation
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing hybrid search: {str(e)}")


# ============================================================================
# Q&A Endpoint
# ============================================================================

@app.post("/qa/answer", response_model=QAResponse, tags=["Q&A"])
async def generate_answer(request: QARequest):
    """
    Generate an answer to a question about a document.

    Uses an LLM (Gemini or Cohere) to answer questions with citations from
    the indexed document database.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not request.document_content:
        raise HTTPException(
            status_code=400,
            detail="document_content is required"
        )

    try:
        # Initialize LLM based on provider
        if request.llm_provider.lower() == "gemini":
            llm = GeminiLLM(
                api_key=request.api_key,
                db=db,
                model=request.model,
                temperature=request.temperature
            )
        elif request.llm_provider.lower() == "cohere":
            llm = CohereLLM(
                api_key=request.api_key,
                db=db,
                model=request.model,
                temperature=request.temperature
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported LLM provider: {request.llm_provider}"
            )

        # Generate answer
        response = llm.generate_answer(request.question, request.document_content)

        # Extract cited documents
        cited_docs = []
        if response.get('cited_docs'):
            for doc in response['cited_docs']:
                cited_docs.append(CitedDocument(
                    doc_id=doc.get('doc_id', ''),
                    title=doc.get('title', ''),
                    content=doc.get('content', '')
                ))

        return QAResponse(
            question=request.question,
            answer=response.get('answer', ''),
            cited_documents=cited_docs,
            thinking=response.get('thinking')
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# ============================================================================
# PDF Processing Endpoints
# ============================================================================

@app.post("/pdf/extract-summary", response_model=PDFExtractionResponse, tags=["PDF Processing"])
async def extract_summary(file: UploadFile = File(..., description="PDF file to process")):
    """
    Extract title and summary from a PDF document.

    Uses LLM to generate a concise summary of the paper.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Extract summary
        result = summary_extraction(temp_file_path)

        return PDFExtractionResponse(
            title=result.title,
            content=result.summary,
            extraction_type="summary"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting summary: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/pdf/extract-abstract", response_model=PDFExtractionResponse, tags=["PDF Processing"])
async def extract_abstract(file: UploadFile = File(..., description="PDF file to process")):
    """
    Extract title and abstract from a PDF document.

    Attempts to parse the abstract section from the paper.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Extract abstract
        result = abstract_extraction(temp_file_path)

        return PDFExtractionResponse(
            title=result.title,
            content=result.abstract,
            extraction_type="abstract"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting abstract: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/pdf/parse-full-text", response_model=PDFExtractionResponse, tags=["PDF Processing"])
async def parse_full_text(file: UploadFile = File(..., description="PDF file to process")):
    """
    Extract full text from a PDF document.

    Returns the complete text content of the PDF.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Extract full text
        full_text = full_text_parse(temp_file_path)

        return PDFExtractionResponse(
            title=Path(file.filename).stem,
            content=full_text,
            extraction_type="full_text"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing full text: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ============================================================================
# Topic Modeling Endpoints
# ============================================================================

@app.post("/topics/analyze-unsupervised", response_model=TopicModelingResponse, tags=["Topic Modeling"])
async def analyze_topics_unsupervised(request: BasicTopicRequest):
    """
    Perform unsupervised topic modeling on all indexed documents.

    Automatically discovers topics in the document collection using BERTopic.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        # Initialize topic clustering
        topic_model = TopicClustering(db=db, api_key=request.api_key)

        # Load documents
        docs, doc_ids, titles = topic_model.load_documents_from_db()

        if len(docs) == 0:
            raise HTTPException(status_code=400, detail="No documents in database")

        # Perform topic modeling
        topic_model.basic_topic_modeling(
            docs=docs,
            doc_ids=doc_ids,
            titles=titles,
            n_topics=request.n_topics,
            min_topic_size=request.min_topic_size
        )

        # Get topic info
        topic_info_df = topic_model.model.get_topic_info()
        topics = []
        for _, row in topic_info_df.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                # Get top keywords for this topic
                topic_words = topic_model.model.get_topic(row['Topic'])
                keywords = [word for word, _ in topic_words[:5]] if topic_words else []

                topics.append(TopicInfo(
                    topic_id=int(row['Topic']),
                    topic_label=row['Name'],
                    count=int(row['Count']),
                    keywords=keywords
                ))

        # Get document-topic assignments
        doc_topics = []
        for i, doc_id in enumerate(doc_ids):
            topic_id = topic_model.model.topics_[i]
            if topic_id != -1:
                topic_info = topic_info_df[topic_info_df['Topic'] == topic_id].iloc[0]
                doc_topics.append(DocumentTopicInfo(
                    doc_id=doc_id,
                    title=titles[i],
                    topic_id=topic_id,
                    topic_label=topic_info['Name'],
                    probability=None
                ))

        return TopicModelingResponse(
            topics=topics,
            document_topics=doc_topics,
            total_documents=len(docs),
            total_topics=len(topics)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in topic modeling: {str(e)}")


@app.post("/topics/analyze-zeroshot", response_model=TopicModelingResponse, tags=["Topic Modeling"])
async def analyze_topics_zeroshot(request: ZeroShotTopicRequest):
    """
    Perform zero-shot topic modeling with predefined topics.

    Classifies documents into user-specified topic categories.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not request.topics or len(request.topics) == 0:
        raise HTTPException(status_code=400, detail="At least one topic must be provided")

    try:
        # Initialize topic clustering
        topic_model = TopicClustering(db=db, api_key=request.api_key)

        # Load documents
        docs, doc_ids, titles = topic_model.load_documents_from_db()

        if len(docs) == 0:
            raise HTTPException(status_code=400, detail="No documents in database")

        # Perform zero-shot topic modeling
        topic_model.zero_shot_topic_modeling(
            docs=docs,
            doc_ids=doc_ids,
            titles=titles,
            topics=request.topics,
            min_topic_size=request.min_topic_size
        )

        # Get topic info
        topic_info_df = topic_model.model.get_topic_info()
        topics = []
        for _, row in topic_info_df.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                # Get top keywords for this topic
                topic_words = topic_model.model.get_topic(row['Topic'])
                keywords = [word for word, _ in topic_words[:5]] if topic_words else []

                topics.append(TopicInfo(
                    topic_id=int(row['Topic']),
                    topic_label=row['Name'],
                    count=int(row['Count']),
                    keywords=keywords
                ))

        # Get document-topic assignments
        doc_topics = []
        for i, doc_id in enumerate(doc_ids):
            topic_id = topic_model.model.topics_[i]
            if topic_id != -1:
                topic_info = topic_info_df[topic_info_df['Topic'] == topic_id].iloc[0]
                doc_topics.append(DocumentTopicInfo(
                    doc_id=doc_id,
                    title=titles[i],
                    topic_id=topic_id,
                    topic_label=topic_info['Name'],
                    probability=None
                ))

        return TopicModelingResponse(
            topics=topics,
            document_topics=doc_topics,
            total_documents=len(docs),
            total_topics=len(topics)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in zero-shot topic modeling: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
            status_code=exc.status_code
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
