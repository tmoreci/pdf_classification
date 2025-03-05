import streamlit as st
import os
import sys
from pathlib import Path

# Add the model_scripts directory to the path
sys.path.append("model_scripts")

# Import your modules
from model_scripts.db import DocumentDatabase
from model_scripts.doc_qa import GeminiLLM
from model_scripts.streamlit_utils import st_error_handling

# Import pages
from pages.qa_interface import show_qa_interface
from pages.database_management import show_database_management
from pages.topic_clustering import show_topic_clustering

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
    page = st.radio(
        "Go to", ["Q&A Interface", "Database Management", "Topic Clustering"]
    )

# Ensure data directories exist
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("pages", exist_ok=True)

# Route to the correct page
if page == "Q&A Interface":
    show_qa_interface(initialize_systems, TEMP_DIR)
elif page == "Database Management":
    show_database_management(
        initialize_systems,
        TEMP_DIR,
        MAX_DISPLAY_DOCS=10,
        RESPONSE_PREVIEW_LENGTH=300,
    )
elif page == "Topic Clustering":
    show_topic_clustering(initialize_systems)

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
