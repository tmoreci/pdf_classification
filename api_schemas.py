"""
Pydantic models for FastAPI request/response schemas.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Document Management Schemas
class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    doc_id: str
    title: str
    message: str


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    doc_id: str
    title: str
    summary: str


class DocumentListResponse(BaseModel):
    """Response containing list of documents."""
    documents: List[DocumentInfo]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    doc_id: str
    message: str


# Search Schemas
class SearchRequest(BaseModel):
    """Request for searching documents."""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class HybridSearchRequest(SearchRequest):
    """Request for hybrid search with weights."""
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector search (0-1)")
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for keyword search (0-1)")


class SearchResult(BaseModel):
    """Single search result."""
    doc_id: str
    title: str
    content: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    """Response containing search results."""
    query: str
    results: List[SearchResult]
    total_results: int


# Q&A Schemas
class QARequest(BaseModel):
    """Request for question answering."""
    question: str = Field(..., description="Question to ask about the document")
    document_content: Optional[str] = Field(None, description="Document text content (if not using file)")
    api_key: str = Field(..., description="API key for LLM service (Gemini or Cohere)")
    model: str = Field(default="gemini-2.0-flash-exp", description="Model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for generation")
    llm_provider: str = Field(default="gemini", description="LLM provider: 'gemini' or 'cohere'")


class CitedDocument(BaseModel):
    """Document cited in the answer."""
    doc_id: str
    title: str
    content: str


class QAResponse(BaseModel):
    """Response containing answer with citations."""
    question: str
    answer: str
    cited_documents: List[CitedDocument]
    thinking: Optional[str] = None


# PDF Processing Schemas
class PDFExtractionResponse(BaseModel):
    """Response from PDF extraction."""
    title: str
    content: str
    extraction_type: str


# Topic Modeling Schemas
class BasicTopicRequest(BaseModel):
    """Request for basic topic modeling."""
    n_topics: Optional[int] = Field(None, description="Number of topics (None for auto-detection)")
    min_topic_size: int = Field(default=5, ge=2, description="Minimum documents per topic")
    api_key: str = Field(..., description="API key for embedding model")


class ZeroShotTopicRequest(BaseModel):
    """Request for zero-shot topic modeling."""
    topics: List[str] = Field(..., description="List of predefined topics")
    min_topic_size: int = Field(default=5, ge=2, description="Minimum documents per topic")
    api_key: str = Field(..., description="API key for embedding model")


class TopicInfo(BaseModel):
    """Information about a topic."""
    topic_id: int
    topic_label: str
    count: int
    keywords: List[str]


class DocumentTopicInfo(BaseModel):
    """Document-topic assignment."""
    doc_id: str
    title: str
    topic_id: int
    topic_label: str
    probability: Optional[float] = None


class TopicModelingResponse(BaseModel):
    """Response from topic modeling."""
    topics: List[TopicInfo]
    document_topics: List[DocumentTopicInfo]
    total_documents: int
    total_topics: int


# System Schemas
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_accessible: bool
    version: str


class StatsResponse(BaseModel):
    """Database statistics response."""
    total_documents: int
    collection_name: str
    embedding_model: str
    database_path: str


# Error Response
class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
