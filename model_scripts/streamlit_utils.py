"""
Utilities for streamlining Streamlit integration with the existing model scripts
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from functools import wraps
from contextlib import contextmanager


@contextmanager
def st_error_handling(error_message="An error occurred"):
    """Context manager for handling errors in Streamlit with appropriate UI feedback"""
    try:
        yield
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        st.exception(e)
        raise


def temporary_file_handler(uploaded_file, directory="data/temp", suffix=None):
    """
    Handles writing an uploaded file to a temporary location and returns the path

    Args:
        uploaded_file: The Streamlit uploaded file
        directory: Directory to store the temporary file
        suffix: File suffix to use (defaults to the uploaded file extension)

    Returns:
        Path to the temporary file
    """
    if suffix is None:
        # Extract suffix from the filename
        _, ext = os.path.splitext(uploaded_file.name)
        suffix = ext if ext else ".tmp"

    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=directory
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def format_citations(response_text, cited_docs, retrieved_docs):
    """
    Format the response text with clickable citation links

    Args:
        response_text: Original response text with [i] citations
        cited_docs: List of cited document indices
        retrieved_docs: Dictionary of retrieved documents

    Returns:
        Formatted text with citation details
    """
    formatted_text = response_text
    citation_details = []

    # Add citation details
    for doc_idx in cited_docs:
        if 0 <= doc_idx < len(retrieved_docs["documents"][0]):
            doc = retrieved_docs["documents"][0][doc_idx]
            # Extract title if available in metadata
            title = "Unknown Source"
            if (
                retrieved_docs.get("metadatas")
                and len(retrieved_docs["metadatas"][0]) > doc_idx
            ):
                metadata = retrieved_docs["metadatas"][0][doc_idx]
                title = metadata.get("title", title)

            # Add to citation details
            citation_details.append(
                f"**[{doc_idx}]** {title}\n\n{doc[:300]}..."
            )

    return formatted_text, citation_details


def cache_data(func):
    """Decorator for caching function outputs in session state"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        key = f"cache_{func.__name__}_{str(args)}_{str(kwargs)}"

        # Check if result is in session state
        if key not in st.session_state:
            # Execute function and store result
            result = func(*args, **kwargs)
            st.session_state[key] = result

        return st.session_state[key]

    return wrapper
