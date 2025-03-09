import streamlit as st
import os
import base64
from model_scripts.streamlit_utils import (
    st_error_handling,
    temporary_file_handler,
)


def show_qa_interface(initialize_systems, TEMP_DIR):
    """
    Display the Q&A Interface page.

    Args:
        initialize_systems: Function to initialize database and LLM systems
        TEMP_DIR: Directory path for storing temporary files
    """
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
                # ! To DO: Figure out why pdf doesn't display sometimes
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
                                "response": model_response,
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
                                            f"{doc[:300]}{'...' if len(doc) > 300 else ''}"
                                        )
