import streamlit as st
import os
import tempfile
from pathlib import Path
import sys
import base64
import pandas as pd

# Add the model_scripts directory to the path
sys.path.append("model_scripts")

# Import your modules
from model_scripts.db import DocumentDatabase
from model_scripts.doc_qa import GeminiLLM
from model_scripts.pdf_utils import summary_extraction, full_text_parse
from model_scripts.streamlit_utils import (
    st_error_handling,
    temporary_file_handler,
    format_citations,
    cache_data,
)

# Configuration settings
# App settings
APP_TITLE = "PDF Research Assistant"
APP_ICON = "📚"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Database settings
DB_PATH = "data/chroma_db"
COLLECTION_NAME = "academic_papers"
TEMP_DIR = "data/temp"
MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding model name

# LLM settings
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.1

# UI settings
MAX_DISPLAY_DOCS = (
    10  # Maximum number of documents to display in the database view
)
RESPONSE_PREVIEW_LENGTH = (
    300  # Number of characters to show in document previews
)

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)

# Initialize session state variables if they don't exist
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "db" not in st.session_state:
    st.session_state.db = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "responses" not in st.session_state:
    st.session_state.responses = []


# Function to initialize the database and LLM
def initialize_systems():
    if st.session_state.gemini_api_key:
        with st_error_handling("Error initializing systems"):
            st.session_state.db = DocumentDatabase(
                gemini_api=st.session_state.gemini_api_key,
                db_path=DB_PATH,
                collection_name=COLLECTION_NAME,
                model_name=MODEL_NAME,
            )
            st.session_state.llm = GeminiLLM(
                api_key=st.session_state.gemini_api_key,
                database=st.session_state.db,
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
            )
            return True
    else:
        st.error("Please enter your Gemini API key")
        return False


# Sidebar for API key input
with st.sidebar:
    st.title("📚 Research Assistant")

    # API key input
    gemini_api_key = st.text_input(
        "Enter your Gemini API key:",
        value=st.session_state.gemini_api_key,
        type="password",
    )

    if gemini_api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = gemini_api_key
        if gemini_api_key:
            if initialize_systems():
                st.success("API key set successfully!")

    # Navigation
    st.subheader("Navigation")
    page = st.radio("Go to", ["Q&A Interface", "Database Management"])

# Ensure data directories exist
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Q&A Interface
if page == "Q&A Interface":
    st.title("PDF Q&A Interface")

    if not st.session_state.gemini_api_key:
        st.warning(
            "Please enter your Gemini API key in the sidebar to continue."
        )
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            # PDF upload
            uploaded_file = st.file_uploader(
                "Upload a PDF to analyze", type=["pdf"]
            )

            if uploaded_file and (
                st.session_state.current_pdf != uploaded_file.name
            ):
                with st_error_handling("Error uploading PDF"):
                    # Save the uploaded file to a temporary location
                    st.session_state.pdf_path = temporary_file_handler(
                        uploaded_file, directory=TEMP_DIR, suffix=".pdf"
                    )

                    st.session_state.current_pdf = uploaded_file.name
                    st.success(f"PDF uploaded: {uploaded_file.name}")
                    st.session_state.responses = []  # Clear previous responses

        # Display PDF preview in the second column if available
        with col2:
            if st.session_state.pdf_path and os.path.exists(
                st.session_state.pdf_path
            ):
                st.subheader("PDF Preview")
                # Display PDF using PDF display hack
                with open(st.session_state.pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

        # Query input
        if st.session_state.pdf_path:
            st.subheader("Ask a question about the PDF")
            query = st.text_area("Your question:", height=100)

            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.button(
                    "Submit Question", use_container_width=True
                )
            with col2:
                if submit_button and not query:
                    st.warning("Please enter a question first.")

            if submit_button and query:
                with st.spinner("Analyzing your query..."):
                    with st_error_handling("Error processing query"):
                        model_response, cited_docs, retrieved_docs = (
                            st.session_state.llm.generate_answer(
                                query, st.session_state.pdf_path
                            )
                        )

                        # Store the response
                        st.session_state.responses.append(
                            {
                                "query": query,
                                "response": model_response.text,
                                "cited_docs": cited_docs,
                                "retrieved_docs": retrieved_docs,
                                "timestamp": st.session_state.get(
                                    "last_run_id", 0
                                ),
                            }
                        )
                        # Update last run ID to force new state
                        st.session_state["last_run_id"] = (
                            st.session_state.get("last_run_id", 0) + 1
                        )

            # Display responses
            if st.session_state.responses:
                st.subheader("Responses")
                for i, resp in enumerate(st.session_state.responses):
                    with st.expander(
                        f"Q: {resp['query'][:50]}{'...' if len(resp['query']) > 50 else ''}",
                        expanded=(i == len(st.session_state.responses) - 1),
                    ):
                        # Format response with citations
                        st.markdown(resp["response"])

                        # Display citations if available
                        if resp["cited_docs"]:
                            st.markdown("---")
                            st.subheader("Cited Sources")

                            for doc_idx in resp["cited_docs"]:
                                if (
                                    0
                                    <= doc_idx
                                    < len(
                                        resp["retrieved_docs"]["documents"][0]
                                    )
                                ):
                                    doc = resp["retrieved_docs"]["documents"][
                                        0
                                    ][doc_idx]
                                    metadata = {}

                                    # Get metadata if available
                                    if (
                                        resp["retrieved_docs"].get("metadatas")
                                        and len(
                                            resp["retrieved_docs"]["metadatas"][
                                                0
                                            ]
                                        )
                                        > doc_idx
                                    ):
                                        metadata = resp["retrieved_docs"][
                                            "metadatas"
                                        ][0][doc_idx]

                                    # Create citation box
                                    with st.container(border=True):
                                        st.markdown(f"**Citation [{doc_idx}]**")
                                        if metadata.get("title"):
                                            st.markdown(
                                                f"**Title:** {metadata.get('title')}"
                                            )

                                        # Show preview of the document
                                        st.markdown("**Preview:**")
                                        st.markdown(
                                            f"{doc[:RESPONSE_PREVIEW_LENGTH]}{'...' if len(doc) > RESPONSE_PREVIEW_LENGTH else ''}"
                                        )

# Database Management
elif page == "Database Management":
    st.title("Database Management")

    if not st.session_state.gemini_api_key:
        st.warning(
            "Please enter your Gemini API key in the sidebar to continue."
        )
    else:
        st.write(
            "Upload PDFs to add to the document database for later retrieval."
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            uploaded_files = st.file_uploader(
                "Upload PDF files", type=["pdf"], accept_multiple_files=True
            )

        with col2:
            if uploaded_files:
                add_button = st.button(
                    "Add to Database", use_container_width=True, type="primary"
                )
            else:
                st.button(
                    "Add to Database", use_container_width=True, disabled=True
                )

        if uploaded_files and add_button:
            with st.container(border=True):
                st.subheader("Processing Documents")
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_log = []
                success_count = 0

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(
                        f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}"
                    )

                    try:
                        # Save the file temporarily
                        file_path = temporary_file_handler(
                            uploaded_file, directory=TEMP_DIR, suffix=".pdf"
                        )

                        # Create a unique document ID
                        doc_id = f"doc_{hash(uploaded_file.name)}_{i}"

                        # Add to database
                        with st_error_handling():
                            st.session_state.db.add_document(file_path, doc_id)
                            success_count += 1

                        # Clean up temporary file
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass

                    except Exception as e:
                        error_log.append(
                            f"Error with '{uploaded_file.name}': {str(e)}"
                        )

                    # Update progress regardless of success/failure
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                # Show final status
                if success_count == len(uploaded_files):
                    st.success(
                        f"Successfully added all {len(uploaded_files)} documents to the database."
                    )
                else:
                    st.warning(
                        f"Added {success_count} out of {len(uploaded_files)} documents to the database."
                    )

                    if error_log:
                        with st.expander("View Error Details"):
                            for error in error_log:
                                st.error(error)

        # Database stats and search
        st.markdown("---")
        st.subheader("Database Explorer")

        if st.session_state.db:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Document List", "Search Database"])

            # Document List Tab
            with tab1:
                with st_error_handling("Error fetching database statistics"):
                    # Get collection info
                    collection_info = st.session_state.db.collection.get()
                    doc_count = (
                        len(collection_info["ids"])
                        if "ids" in collection_info
                        else 0
                    )

                    st.metric("Total Documents", doc_count)

                    if doc_count > 0:
                        if (
                            collection_info.get("metadatas")
                            and len(collection_info["metadatas"]) > 0
                        ):
                            # Prepare data for the table
                            data = []
                            for i, metadata in enumerate(
                                collection_info["metadatas"]
                            ):
                                if i < MAX_DISPLAY_DOCS:  # Limit display
                                    title = metadata.get(
                                        "title", "Unknown title"
                                    )
                                    doc_id = collection_info["ids"][i]
                                    data.append(
                                        {
                                            "ID": doc_id,
                                            "Title": title,
                                            "Preview": collection_info[
                                                "documents"
                                            ][i][:100]
                                            + "...",
                                        }
                                    )

                            # Create and display the dataframe
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df, use_container_width=True)

                                if doc_count > MAX_DISPLAY_DOCS:
                                    st.info(
                                        f"Displaying {MAX_DISPLAY_DOCS} of {doc_count} documents. Use the search tab to find specific documents."
                                    )
                    else:
                        st.info(
                            "No documents in the database yet. Upload PDF files to add them."
                        )

            # Search Tab
            with tab2:
                search_query = st.text_input(
                    "Search database for documents:",
                    placeholder="Enter keywords to search",
                )

                if search_query:
                    with st.spinner("Searching database..."):
                        with st_error_handling("Error searching the database"):
                            results = st.session_state.db.vector_search(
                                search_query
                            )

                            if (
                                results
                                and results.get("documents")
                                and len(results["documents"][0]) > 0
                            ):
                                st.success(
                                    f"Found {len(results['documents'][0])} matching documents"
                                )

                                for i, (doc, metadata) in enumerate(
                                    zip(
                                        results["documents"][0],
                                        results["metadatas"][0],
                                    )
                                ):
                                    with st.expander(
                                        f"Result {i+1}: {metadata.get('title', 'Unknown document')}"
                                    ):
                                        st.markdown(
                                            f"**Document ID:** {results['ids'][0][i]}"
                                        )
                                        st.markdown("**Preview:**")
                                        st.markdown(
                                            f"{doc[:RESPONSE_PREVIEW_LENGTH]}{'...' if len(doc) > RESPONSE_PREVIEW_LENGTH else ''}"
                                        )
                            else:
                                st.info(
                                    "No matching documents found. Try different search terms."
                                )

# Footer
st.sidebar.markdown("---")

# Add some system information
with st.sidebar.expander("System Information"):
    st.write(f"**Database Path:** {DB_PATH}")
    st.write(f"**Collection Name:** {COLLECTION_NAME}")
    st.write(f"**Embedding Model:** {MODEL_NAME}")
    st.write(f"**LLM Model:** {DEFAULT_MODEL}")

    # Technical details for debugging
    if st.session_state.db:
        try:
            collection_info = st.session_state.db.collection.get()
            st.write(
                f"**Index Size:** {len(collection_info['ids']) if 'ids' in collection_info else 0} documents"
            )
        except Exception:
            st.write("**Index Size:** Unable to retrieve")
