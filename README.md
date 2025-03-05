# Research Paper Q&A Assistant

A powerful tool for researchers to analyze, search, and extract insights from academic papers. This application combines vector search, LLM-powered question answering, and topic clustering to provide an enhanced research experience.

## Features

- **Document Q&A**: Upload PDF papers and ask complex questions about their content
- **Context-Aware Responses**: Get answers based on the paper's content with optional references to related research
- **Hybrid Search**: Find relevant documents using both semantic and keyword search capabilities
- **Topic Clustering**: Discover patterns and themes across your research collection
- **Document Management**: Easily add, index, and organize your research papers

## System Architecture

The application is built with a modular architecture:

```
├── app.py                  # Main Streamlit application entry point
├── run.py                  # Script to initialize and run the application
├── model_scripts/          # Core functionality modules
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
   git clone https://github.com/yourusername/pdf-research-assistant.git
   cd pdf-research-assistant
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

Run the application using the provided script:

```bash
python run.py
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

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
- **PDF Processing**: PyMuPDF for document parsing and extraction

### Search Capabilities

The system uses a hybrid search approach that combines:

- **Vector Search**: For semantic similarity using document embeddings
- **Keyword Search**: Using BM25 algorithm for traditional search
- **Weighted Ranking**: Combines both approaches for optimal results

## Requirements

The application requires the following Python packages:

```
streamlit
chromadb
sentence-transformers
pymupdf
langchain
langchain-community
google-generativeai
bertopic
pydantic
tqdm
rank_bm25
jinja2
```
## Acknowledgments

- [Streamlit](https://streamlit.io/) for the interactive web interface
- [ChromaDB](https://www.trychroma.com/) for vector database capabilities
- [Google Gemini API](https://ai.google.dev/) for LLM integration
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling