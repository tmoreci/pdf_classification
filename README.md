# Research Paper Q&A Assistant
The main functionality of this tool is for Q&A over an uploaded PDF with a local vector database storing summaries of additional research papers that can be used in analysis. The tool is powered by Gemini-2.0-Flash, which is provided a tool for querying the database for relevant summaries and is prompted to cite these summaries as needed. The retrieval method is a combination of cosine similarity search on the embeddings and BM25 on the actual summaries, inspired by Anthropic's work on contextual document embeddings. I've also added some topic modelling analysis using the BERTopic package to help get a better sense of what documents you can actually retrieve from your local database.

**NEW**: The application now includes a comprehensive **FastAPI REST API** for easy integration with any frontend framework!

## Features

- **Document Q&A**: Upload PDF papers and ask questions about their content
- **Context-Aware Responses**: Get answers based on the paper's content with optional references to stored summaries of other research papers
- **Hybrid Search**: Find relevant documents using both semantic and keyword search capabilities
- **Topic Clustering**: Discover patterns and themes across your research collection
- **Document Management**: Easily add, index, and organize your research papers
- **RESTful API**: FastAPI-based endpoints for integration with modern frontends (React, Vue, Angular, etc.)

## System Architecture

The application is built with a modular architecture supporting both Streamlit UI and FastAPI:

```
├── app.py                  # Main Streamlit application entry point
├── run.py                  # Script to initialize and run Streamlit app
├── api.py                  # FastAPI application with REST endpoints
├── api_schemas.py          # Pydantic models for API requests/responses
├── run_api.py              # Script to run the FastAPI server
├── model_scripts/          # Core functionality modules (shared between UI and API)
│   ├── base.py             # Abstract base classes and data models
│   ├── db.py               # Document database and vector store operations
│   ├── doc_qa.py           # LLM-based question answering
│   ├── pdf_utils.py        # PDF processing utilities
│   ├── prompts.py          # LLM prompt templates
│   ├── streamlit_utils.py  # Streamlit UI helper functions
│   └── topic_clustering.py # Topic modeling and clustering
└── pages/                  # Streamlit UI pages
    ├── database_management.py
    ├── qa_interface.py
    └── topic_clustering.py
```

## Getting Started

### Prerequisites

- Python 3.9+
- Gemini API key ([Get one here](https://ai.google.dev/))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tmoreci/pdf_qa
   cd pdf_qa
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data/chroma_db data/temp pages
   ```

### Running the Application

You can run either the Streamlit UI or the FastAPI server (or both simultaneously on different ports).

#### Option 1: Streamlit UI (Web Interface)

Run the Streamlit application using the provided script:

```bash
python run.py
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

#### Option 2: FastAPI Server (REST API)

Run the FastAPI server for programmatic access:

```bash
python run_api.py
```

Or directly with uvicorn:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Then access:
- Interactive API docs (Swagger): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc
- API endpoints: http://localhost:8000/

See [API_GUIDE.md](API_GUIDE.md) for complete API documentation and usage examples.

## Usage Guide

### Q&A Interface

1. Enter your Gemini API key in the sidebar
2. Upload a research paper PDF
3. Type your question in the input field
4. View the response, which may include citations to external sources when relevant

### Database Management

1. Upload new papers to add to your knowledge base
2. Search through existing papers
3. Manage and organize your paper collection

### Topic Clustering

1. Analyze patterns across your document collection
2. Discover emergent topics and themes
3. Visualize relationships between research areas

## Technical Details

### Components

- **Vector Database**: ChromaDB for efficient semantic search
- **Text Embeddings**: Sentence transformers for document vectorization
- **LLM Integration**: Google Gemini API for question answering
- **Topic Modeling**: BERTopic for document clustering and theme discovery
- **PDF Processing**: PyMuPDF for document parsing and extraction, Gemini for abstract and summary extraction

### Search Capabilities

The system uses a hybrid search approach that combines:

- **Vector Search**: For semantic similarity using document embeddings
- **Keyword Search**: Using BM25 algorithm for traditional search
- **Weighted Ranking**: Combines both approaches for optimal results

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the interactive web interface
- [ChromaDB](https://www.trychroma.com/) for vector database capabilities
- [Google Gemini API](https://ai.google.dev/) for LLM integration
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling