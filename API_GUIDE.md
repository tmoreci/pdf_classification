# PDF Q&A FastAPI Guide

This guide explains how to use the FastAPI endpoints for the PDF Q&A system.

## Table of Contents
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
  - [System Endpoints](#system-endpoints)
  - [Document Management](#document-management)
  - [Search Operations](#search-operations)
  - [Question Answering](#question-answering)
  - [PDF Processing](#pdf-processing)
  - [Topic Modeling](#topic-modeling)
- [Usage Examples](#usage-examples)
- [Frontend Integration](#frontend-integration)

## Getting Started

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the API Server

Start the FastAPI server:

```bash
# Development mode with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### System Endpoints

#### Health Check
```http
GET /health
```

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "database_accessible": true,
  "version": "1.0.0"
}
```

#### Database Statistics
```http
GET /stats
```

Get statistics about the document database.

**Response:**
```json
{
  "total_documents": 42,
  "collection_name": "academic_papers",
  "embedding_model": "all-MiniLM-L6-v2",
  "database_path": "data/chroma_db"
}
```

---

### Document Management

#### Upload Document
```http
POST /documents/upload
```

Upload and index a PDF document.

**Request:**
- Content-Type: `multipart/form-data`
- `file`: PDF file (required)
- `doc_id`: Custom document ID (optional)

**Response:**
```json
{
  "doc_id": "paper_2024",
  "title": "Attention Is All You Need",
  "message": "Document uploaded and indexed successfully"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@paper.pdf" \
  -F "doc_id=attention_paper"
```

#### List Documents
```http
GET /documents/list
```

Get all indexed documents.

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "paper_2024",
      "title": "Attention Is All You Need",
      "summary": "This paper introduces the Transformer architecture..."
    }
  ],
  "total_count": 1
}
```

#### Get Document
```http
GET /documents/{doc_id}
```

Get details of a specific document.

**Response:**
```json
{
  "doc_id": "paper_2024",
  "title": "Attention Is All You Need",
  "summary": "This paper introduces the Transformer architecture..."
}
```

#### Delete Document
```http
DELETE /documents/{doc_id}
```

Remove a document from the database.

**Response:**
```json
{
  "doc_id": "paper_2024",
  "message": "Document deleted successfully"
}
```

---

### Search Operations

#### Vector Search
```http
POST /search/vector
```

Perform semantic vector search.

**Request:**
```json
{
  "query": "transformer architecture for NLP",
  "n_results": 5
}
```

**Response:**
```json
{
  "query": "transformer architecture for NLP",
  "results": [
    {
      "doc_id": "paper_2024",
      "title": "Attention Is All You Need",
      "content": "Summary text...",
      "score": 0.89
    }
  ],
  "total_results": 5
}
```

#### Keyword Search
```http
POST /search/keyword
```

Perform BM25 keyword search.

**Request:**
```json
{
  "query": "attention mechanism",
  "n_results": 5
}
```

#### Hybrid Search
```http
POST /search/hybrid
```

Combine vector and keyword search.

**Request:**
```json
{
  "query": "neural machine translation",
  "n_results": 5,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

---

### Question Answering

#### Generate Answer
```http
POST /qa/answer
```

Ask questions about a document with LLM-powered answers.

**Request:**
```json
{
  "question": "What is the main contribution of this paper?",
  "document_content": "Full text of the paper...",
  "api_key": "your_gemini_api_key",
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.1,
  "llm_provider": "gemini"
}
```

**Response:**
```json
{
  "question": "What is the main contribution of this paper?",
  "answer": "The main contribution is the Transformer architecture...",
  "cited_documents": [
    {
      "doc_id": "related_paper_1",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "content": "Summary..."
    }
  ],
  "thinking": "Let me analyze the paper's key contributions..."
}
```

**Supported LLM Providers:**
- `gemini`: Google Gemini API
- `cohere`: Cohere API

---

### PDF Processing

#### Extract Summary
```http
POST /pdf/extract-summary
```

Extract title and LLM-generated summary from PDF.

**Request:**
- Content-Type: `multipart/form-data`
- `file`: PDF file

**Response:**
```json
{
  "title": "Attention Is All You Need",
  "content": "This paper introduces the Transformer...",
  "extraction_type": "summary"
}
```

#### Extract Abstract
```http
POST /pdf/extract-abstract
```

Extract title and abstract from PDF.

**Request:**
- Content-Type: `multipart/form-data`
- `file`: PDF file

**Response:**
```json
{
  "title": "Attention Is All You Need",
  "content": "The dominant sequence transduction models...",
  "extraction_type": "abstract"
}
```

#### Parse Full Text
```http
POST /pdf/parse-full-text
```

Extract complete text from PDF.

**Request:**
- Content-Type: `multipart/form-data`
- `file`: PDF file

**Response:**
```json
{
  "title": "paper",
  "content": "Full text content of the PDF...",
  "extraction_type": "full_text"
}
```

---

### Topic Modeling

#### Unsupervised Topic Analysis
```http
POST /topics/analyze-unsupervised
```

Automatically discover topics in document collection.

**Request:**
```json
{
  "n_topics": 10,
  "min_topic_size": 5,
  "api_key": "your_gemini_api_key"
}
```

**Response:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "topic_label": "0_transformer_attention_neural_network",
      "count": 15,
      "keywords": ["transformer", "attention", "neural", "network", "deep"]
    }
  ],
  "document_topics": [
    {
      "doc_id": "paper_2024",
      "title": "Attention Is All You Need",
      "topic_id": 0,
      "topic_label": "0_transformer_attention_neural_network",
      "probability": null
    }
  ],
  "total_documents": 42,
  "total_topics": 10
}
```

#### Zero-Shot Topic Analysis
```http
POST /topics/analyze-zeroshot
```

Classify documents into predefined topics.

**Request:**
```json
{
  "topics": [
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning"
  ],
  "min_topic_size": 5,
  "api_key": "your_gemini_api_key"
}
```

---

## Usage Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload a document
with open("paper.pdf", "rb") as f:
    files = {"file": f}
    data = {"doc_id": "my_paper"}
    response = requests.post(f"{BASE_URL}/documents/upload", files=files, data=data)
    print(response.json())

# Search for documents
search_request = {
    "query": "transformer architecture",
    "n_results": 5
}
response = requests.post(f"{BASE_URL}/search/hybrid", json=search_request)
results = response.json()
print(f"Found {results['total_results']} results")

# Ask a question
qa_request = {
    "question": "What is the main contribution?",
    "document_content": "Full paper text...",
    "api_key": "your_api_key",
    "llm_provider": "gemini"
}
response = requests.post(f"{BASE_URL}/qa/answer", json=qa_request)
answer = response.json()
print(f"Answer: {answer['answer']}")
```

### JavaScript/TypeScript Example

```typescript
const BASE_URL = "http://localhost:8000";

// Upload a document
async function uploadDocument(file: File, docId: string) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("doc_id", docId);

  const response = await fetch(`${BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });

  return await response.json();
}

// Search documents
async function searchDocuments(query: string) {
  const response = await fetch(`${BASE_URL}/search/hybrid`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: query,
      n_results: 5,
      vector_weight: 0.7,
      keyword_weight: 0.3,
    }),
  });

  return await response.json();
}

// Ask a question
async function askQuestion(question: string, documentContent: string, apiKey: string) {
  const response = await fetch(`${BASE_URL}/qa/answer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: question,
      document_content: documentContent,
      api_key: apiKey,
      llm_provider: "gemini",
    }),
  });

  return await response.json();
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@paper.pdf" \
  -F "doc_id=my_paper"

# List documents
curl http://localhost:8000/documents/list

# Search documents
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer architecture",
    "n_results": 5,
    "vector_weight": 0.7,
    "keyword_weight": 0.3
  }'

# Ask a question
curl -X POST "http://localhost:8000/qa/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main contribution?",
    "document_content": "Full paper text...",
    "api_key": "your_api_key",
    "llm_provider": "gemini"
  }'
```

---

## Frontend Integration

### Recommended Architecture

For a production frontend application:

1. **Document Upload Flow**
   - Upload PDF → `/documents/upload`
   - Display uploaded documents → `/documents/list`
   - Allow deletion → `/documents/{doc_id}` (DELETE)

2. **Search Interface**
   - Implement search bar with `/search/hybrid` endpoint
   - Display results with titles, snippets, and scores
   - Allow clicking results to view full document

3. **Q&A Interface**
   - Upload/paste document text
   - Input question field
   - Display answer with cited sources
   - Show "thinking" process (optional)

4. **Topic Visualization**
   - Load all documents → `/documents/list`
   - Analyze topics → `/topics/analyze-unsupervised` or `/topics/analyze-zeroshot`
   - Display topic distribution charts
   - Filter documents by topic

### State Management Recommendations

```typescript
interface AppState {
  documents: Document[];
  searchResults: SearchResult[];
  currentAnswer: QAResponse | null;
  topics: TopicInfo[];
  loading: boolean;
  error: string | null;
}

// Actions
async function uploadDocument(file: File): Promise<void>
async function searchDocuments(query: string): Promise<void>
async function askQuestion(question: string, content: string): Promise<void>
async function analyzeTopics(): Promise<void>
```

### Error Handling

All endpoints return errors in the following format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

Always handle errors gracefully in your frontend:

```typescript
try {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error);
  }
  return await response.json();
} catch (error) {
  console.error("API Error:", error);
  // Show user-friendly error message
}
```

### CORS Configuration

The API has CORS enabled with `allow_origins=["*"]` for development. In production, update this in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## Performance Considerations

1. **File Upload Limits**: Configure maximum file size in production
2. **Rate Limiting**: Consider adding rate limiting for API endpoints
3. **Caching**: Implement caching for frequently accessed documents
4. **Background Processing**: Use background tasks for large document uploads
5. **Database Scaling**: Monitor ChromaDB performance with large collections

## Security Best Practices

1. **API Keys**: Never expose API keys in frontend code
2. **Authentication**: Add authentication middleware for production
3. **Input Validation**: All inputs are validated by Pydantic models
4. **File Uploads**: Only PDF files are accepted
5. **HTTPS**: Always use HTTPS in production

## Next Steps

1. Set up your frontend framework (React, Vue, Angular, etc.)
2. Create API client service
3. Implement UI components for each feature
4. Add authentication and authorization
5. Deploy API and frontend separately or together

For more information, see the main [README.md](README.md) file.
