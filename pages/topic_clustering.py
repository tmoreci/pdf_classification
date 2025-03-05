import streamlit as st
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
from model_scripts.topic_clustering import TopicClustering
from model_scripts.streamlit_utils import st_error_handling


def show_topic_clustering(initialize_systems):
    """
    Display the Topic Clustering page.

    Args:
        initialize_systems: Function to initialize database and LLM systems
    """
    st.title("Document Topic Clustering")

    if not st.session_state.gemini_api_key:
        st.warning(
            "Please enter your Gemini API key in the sidebar to continue."
        )
    else:
        # Initialize systems if needed
        if "db" not in st.session_state or st.session_state.db is None:
            initialize_systems()

        st.write(
            """
        This interface allows you to discover topics in your document collection using two methods:
        1. **Unsupervised Clustering**: Automatically finds topics in your documents
        2. **Zero-Shot Topic Modeling**: Categorizes documents according to topics you define
        """
        )

        # Check if we have documents in the database
        with st_error_handling("Error accessing the database"):
            collection_info = st.session_state.db.collection.get()
            doc_count = (
                len(collection_info["ids"]) if "ids" in collection_info else 0
            )

        if doc_count == 0:
            st.warning(
                "No documents found in the database. Please add documents in the Database Management page before clustering."
            )
        else:
            st.success(f"Found {doc_count} documents in the database.")

            # Create tabs for different clustering methods
            clustering_method = st.radio(
                "Select Clustering Method:",
                ["Unsupervised Topic Modeling", "Zero-Shot Topic Modeling"],
            )

            # Initialize TopicClustering when needed
            if "topic_clustering" not in st.session_state:
                st.session_state.topic_clustering = None

            # Common settings
            col1, col2 = st.columns(2)
            with col1:
                min_topic_size = st.slider(
                    "Minimum Topic Size:",
                    1,
                    10,
                    3,
                    help="Minimum number of documents required to form a topic",
                )

            with col2:
                if clustering_method == "Unsupervised Topic Modeling":
                    nr_topics_options = ["auto", "5", "10", "15", "20"]
                    nr_topics = st.selectbox(
                        "Number of Topics:",
                        nr_topics_options,
                        help="'auto' lets the algorithm decide the optimal number",
                    )
                    # Convert to integer if not 'auto'
                    if nr_topics != "auto":
                        nr_topics = int(nr_topics)

            # Zero-shot specific settings
            if clustering_method == "Zero-Shot Topic Modeling":
                st.subheader("Define Your Topics")
                st.write(
                    "Enter topics that you want to categorize documents into, one per line:"
                )

                topic_text = st.text_area(
                    "Topics:",
                    "Agents\nRetrieval\nTraining\nEvaluation",
                    height=150,
                    help="Enter each topic on a new line",
                )

                topic_list = [
                    t.strip() for t in topic_text.split("\n") if t.strip()
                ]

                min_similarity = st.slider(
                    "Minimum Similarity:",
                    0.1,
                    0.9,
                    0.4,
                    0.05,
                    help="Minimum similarity threshold for assigning documents to topics",
                )

                # Display the topics that will be used
                st.write(
                    f"Will cluster documents into these {len(topic_list)} topics:"
                )
                st.write(", ".join(topic_list))

            # Run clustering button
            if st.button("Run Topic Clustering", type="primary"):
                with st.spinner("Loading documents and computing topics..."):
                    try:
                        # Initialize topic clustering
                        topic_clustering = TopicClustering(st.session_state.db)
                        st.session_state.topic_clustering = topic_clustering

                        # Load documents
                        documents = topic_clustering.load_documents_from_db()

                        if len(documents) < min_topic_size:
                            st.error(
                                f"Not enough documents. You have {len(documents)} documents but need at least {min_topic_size} for the minimum topic size."
                            )
                        else:
                            # Run the appropriate clustering method
                            if (
                                clustering_method
                                == "Unsupervised Topic Modeling"
                            ):
                                with st_error_handling(
                                    "Error during topic modeling"
                                ):
                                    st.info(
                                        f"Running unsupervised clustering with min_topic_size={min_topic_size} and nr_topics={nr_topics}"
                                    )
                                    topics, probs = (
                                        topic_clustering.basic_topic_modeling(
                                            min_topic_size=min_topic_size,
                                            nr_topics=nr_topics,
                                            verbose=False,
                                        )
                                    )

                                    # Store results in session state
                                    st.session_state.topics = topics
                                    st.session_state.probs = probs

                                    # Success message
                                    topic_info = (
                                        topic_clustering.get_topic_info()
                                    )
                                    num_topics = len(
                                        topic_info[topic_info["Topic"] != -1]
                                    )
                                    st.success(
                                        f"Successfully identified {num_topics} topics from {len(documents)} documents"
                                    )

                            else:  # Zero-Shot Topic Modeling
                                with st_error_handling(
                                    "Error during zero-shot topic modeling"
                                ):
                                    st.info(
                                        f"Running zero-shot clustering with {len(topic_list)} defined topics"
                                    )
                                    topics = topic_clustering.zero_shot_topic_modeling(
                                        topic_list=topic_list,
                                        min_similarity=min_similarity,
                                        min_topic_size=min_topic_size,
                                        verbose=False,
                                    )

                                    # Store results in session state
                                    st.session_state.topics = topics

                                    # Success message
                                    topic_info = (
                                        topic_clustering.get_topic_info()
                                    )
                                    num_topics = len(
                                        topic_info[topic_info["Topic"] != -1]
                                    )
                                    st.success(
                                        f"Successfully categorized documents into {num_topics} topics"
                                    )

                            # Set flag to show results
                            st.session_state.show_topic_results = True

                    except Exception as e:
                        st.error(f"Error in topic clustering: {str(e)}")
                        st.exception(e)

            # Display results if available
            if (
                st.session_state.get("show_topic_results", False)
                and st.session_state.topic_clustering
            ):
                st.markdown("---")
                st.subheader("Topic Clustering Results")

                # Create tabs for different views
                results_tab1, results_tab2, results_tab3 = st.tabs(
                    ["Topic Overview", "Document Topics", "Visualization"]
                )

                # Tab 1: Topic Overview
                with results_tab1:
                    try:
                        topic_info = (
                            st.session_state.topic_clustering.get_topic_info()
                        )

                        # Format the DataFrame for display
                        display_info = topic_info.copy()

                        # Handle the special case of -1 (outliers)
                        display_info.loc[
                            display_info["Topic"] == -1, "Name"
                        ] = "Outliers"

                        # Display the topic information
                        st.dataframe(display_info, use_container_width=True)

                        # Summary statistics
                        total_docs = display_info["Count"].sum()
                        outlier_count = (
                            display_info.loc[
                                display_info["Topic"] == -1, "Count"
                            ].values[0]
                            if -1 in display_info["Topic"].values
                            else 0
                        )

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Documents", total_docs)
                        col2.metric(
                            "Categorized Documents", total_docs - outlier_count
                        )
                        col3.metric("Outliers", outlier_count)

                    except Exception as e:
                        st.error(f"Error displaying topic overview: {str(e)}")

                # Tab 2: Document Topics
                with results_tab2:
                    try:
                        doc_info = (
                            st.session_state.topic_clustering.get_document_info()
                        )

                        # Preview of documents and their assigned topics
                        st.write("Documents and their assigned topics:")

                        # Add a preview column with truncated document text
                        doc_info["Preview"] = doc_info["document"].apply(
                            lambda x: x[:200] + "..." if len(x) > 200 else x
                        )

                        # Select columns to display
                        display_columns = ["topic"]

                        # Add metadata columns if available
                        metadata_cols = [
                            col
                            for col in doc_info.columns
                            if col
                            not in [
                                "document",
                                "topic",
                                "probability",
                                "Preview",
                            ]
                        ]
                        display_columns.extend(metadata_cols)
                        display_columns.extend(["Preview"])

                        # Display the DataFrame
                        st.dataframe(
                            doc_info[display_columns], use_container_width=True
                        )

                        # Allow downloading the full document-topic assignments
                        csv = doc_info.to_csv(index=False)
                        st.download_button(
                            label="Download Document Topic Data",
                            data=csv,
                            file_name="document_topics.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"Error displaying document topics: {str(e)}")

                # Tab 3: Visualization
                with results_tab3:
                    try:
                        # For zero-shot topic modeling, use specialized visualization
                        if clustering_method == "Zero-Shot Topic Modeling":
                            st.write("Zero-Shot Topic Distribution:")

                            # Create figure
                            fig = (
                                st.session_state.topic_clustering.visualize_zero_shot_topics()
                            )
                            st.pyplot(fig)

                            # Convert matplotlib figure to bytes for download
                            buf = io.BytesIO()
                            fig.savefig(
                                buf, format="png", dpi=300, bbox_inches="tight"
                            )
                            buf.seek(0)

                            # Add download button
                            st.download_button(
                                label="Download Topic Visualization",
                                data=buf,
                                file_name="topic_distribution.png",
                                mime="image/png",
                            )

                        # For unsupervised modeling, try to use the interactive visualization
                        else:
                            # We need at least 10 documents for the UMAP visualization
                            if len(st.session_state.topics) >= 10:
                                st.write("Topic Visualization:")
                                fig = (
                                    st.session_state.topic_clustering.visualize_topics()
                                )

                                # Check if we got a figure back
                                if fig:
                                    from streamlit.components.v1 import html

                                    # Convert the figure to HTML and display
                                    html_str = fig.to_html()
                                    html(html_str, height=600)
                                else:
                                    st.warning(
                                        "Not enough distinct topics for interactive visualization."
                                    )
                            else:
                                st.warning(
                                    "Not enough documents for the interactive visualization (need at least 10)."
                                )

                            # Fallback to a simpler visualization
                            st.write("Topic Distribution:")

                            # Create a simple bar chart of document counts per topic
                            topic_counts = np.unique(
                                st.session_state.topics, return_counts=True
                            )
                            topic_ids = topic_counts[0]
                            counts = topic_counts[1]

                            # Create DataFrame for plotting
                            topic_df = pd.DataFrame(
                                {
                                    "Topic": [
                                        f"Topic {t}" if t != -1 else "Outliers"
                                        for t in topic_ids
                                    ],
                                    "Count": counts,
                                }
                            )

                            # Sort by topic ID
                            topic_df = topic_df.sort_values("Topic")

                            # Create and display bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(
                                topic_df["Topic"],
                                topic_df["Count"],
                                color=[
                                    "#ff9999" if "Outliers" in t else "#5599ff"
                                    for t in topic_df["Topic"]
                                ],
                            )

                            # Add count labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.1,
                                    f"{height:.0f}",
                                    ha="center",
                                    va="bottom",
                                )

                            ax.set_title("Document Count by Topic")
                            ax.set_xlabel("Topics")
                            ax.set_ylabel("Number of Documents")
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()

                            st.pyplot(fig)

                            # Convert matplotlib figure to bytes for download
                            buf = io.BytesIO()
                            fig.savefig(
                                buf, format="png", dpi=300, bbox_inches="tight"
                            )
                            buf.seek(0)

                            # Add download button
                            st.download_button(
                                label="Download Topic Visualization",
                                data=buf,
                                file_name="topic_distribution.png",
                                mime="image/png",
                            )

                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                        st.exception(e)
