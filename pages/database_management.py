import streamlit as st
import pandas as pd
from model_scripts.streamlit_utils import (
    st_error_handling,
    temporary_file_handler,
)
import os


def show_database_management(
    initialize_systems,
    TEMP_DIR,
    MAX_DISPLAY_DOCS=10,
    RESPONSE_PREVIEW_LENGTH=300,
):
    """
    Display the Database Management page.

    Args:
        initialize_systems: Function to initialize database and LLM systems
        TEMP_DIR: Directory path for storing temporary files
        MAX_DISPLAY_DOCS: Maximum number of documents to display in the database view
        RESPONSE_PREVIEW_LENGTH: Number of characters to show in document previews
    """
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
