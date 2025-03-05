import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model_scripts.topic_clustering import TopicClustering
from model_scripts.streamlit_utils import st_error_handling


def show_topic_clustering(initialize_systems):
    """
    Display the topic clustering interface page.

    This page allows users to analyze document collections using topic clustering techniques.

    Args:
        initialize_systems: Function to initialize database and LLM
    """
    st.title("Document Topic Clustering")

    # Check if systems are initialized
    if not initialize_systems():
        return

    # Show sidebar options
    with st.sidebar:
        st.subheader("Topic Modeling Settings")

        model_type = st.radio(
            "Topic Modeling Approach:",
            ["Unsupervised", "Zero-Shot"],
            help="Choose between unsupervised topic detection or pre-defined topic categories",
        )

        if model_type == "Unsupervised":
            min_topic_size = st.slider(
                "Minimum Topic Size:",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum number of documents required to form a topic",
            )

            nr_topics = st.select_slider(
                "Number of Topics:",
                options=["auto", "5", "10", "20", "30", "50"],
                value="auto",
                help="Number of topics to extract ('auto' lets the algorithm decide)",
            )
            nr_topics = int(nr_topics) if nr_topics != "auto" else "auto"

        else:  # Zero-Shot
            min_topic_size = st.slider(
                "Minimum Topic Size:",
                min_value=1,
                max_value=10,
                value=3,
                help="Minimum number of documents required to form a topic",
            )

            min_similarity = st.slider(
                "Minimum Similarity:",
                min_value=0.1,
                max_value=0.9,
                value=0.4,
                step=0.05,
                help="Minimum similarity threshold for assigning documents to topics",
            )

            # Input for custom topics
            topic_input = st.text_area(
                "Enter Topics (one per line):",
                value="Machine Learning\nReinforcement Learning\nNatural Language Processing\nComputer Vision\nNeural Networks",
                help="Enter predefined topics, one per line",
            )
            custom_topics = [
                topic.strip()
                for topic in topic_input.split("\n")
                if topic.strip()
            ]

    # Main content
    with st.spinner("Loading documents..."):
        topic_clustering = TopicClustering(st.session_state.db)

        try:
            documents = topic_clustering.load_documents_from_db()
            st.write(f"Loaded {len(documents)} documents from the database.")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return

    # Run topic modeling
    if st.button("Run Topic Analysis", type="primary"):
        with st.spinner("Analyzing topics..."):
            try:
                if model_type == "Unsupervised":
                    topics, _ = topic_clustering.basic_topic_modeling(
                        min_topic_size=min_topic_size,
                        nr_topics=nr_topics,
                        verbose=False,
                    )
                    st.session_state.topic_model_type = "unsupervised"
                else:
                    topics, _ = topic_clustering.zero_shot_topic_modeling(
                        topic_list=custom_topics,
                        min_similarity=min_similarity,
                        min_topic_size=min_topic_size,
                        verbose=False,
                    )
                    st.session_state.topic_model_type = "zero-shot"

                # Store in session state for later use
                st.session_state.topic_clustering = topic_clustering

                # Success message
                unique_topics = len(set(topics)) - (
                    1 if -1 in topics else 0
                )  # Exclude outliers
                st.success(
                    f"Topic analysis complete! Found {unique_topics} topics."
                )

            except Exception as e:
                st.error(f"Error in topic modeling: {e}")
                st.exception(e)
                return

    # Display results if topic model exists
    if (
        hasattr(st.session_state, "topic_clustering")
        and st.session_state.topic_clustering.topic_model
    ):
        topic_clustering = st.session_state.topic_clustering

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(
            ["Topic Overview", "Topic Distribution", "Document Details"]
        )

        with tab1:
            st.subheader("Topic Overview")

            # Show topic info table
            topic_info = topic_clustering.get_topic_info()

            # Clean up the display
            display_info = topic_info.copy()
            display_info["Name"] = display_info["Name"].apply(
                lambda x: (
                    ", ".join(x.split("_")[:3]) if x != "-1" else "Outliers"
                )
            )
            display_info = display_info.rename(
                columns={
                    "Topic": "Topic ID",
                    "Count": "Documents",
                    "Name": "Top Words",
                }
            )

            st.dataframe(
                display_info[["Topic ID", "Documents", "Top Words"]],
                hide_index=True,
                use_container_width=True,
            )

            # Topic visualization
            st.subheader("Topic Visualization")

            try:
                if st.session_state.topic_model_type == "unsupervised":
                    fig = topic_clustering.visualize_topics()
                else:
                    fig = topic_clustering.visualize_zero_shot_topics()

                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"Error in visualization: {e}")
                # Try fallback
                try:
                    fig = topic_clustering.visualize_document_distribution()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e2:
                    st.error(f"Fallback visualization also failed: {e2}")

        with tab2:
            st.subheader("Document Distribution by Topic")

            # Document distribution visualization
            try:
                top_n = st.slider("Number of topics to display:", 3, 15, 8)
                fig = topic_clustering.visualize_document_distribution(
                    top_n=top_n
                )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(
                    f"Error creating document distribution visualization: {e}"
                )

        with tab3:
            st.subheader("Document Details")

            # Get document info
            doc_info = topic_clustering.get_document_info()

            # Add topic label column if missing
            if "topic_label" not in doc_info.columns:
                # Create topic labels
                topic_labels = {}
                for topic_id in doc_info["topic"].unique():
                    if topic_id == -1:
                        topic_labels[topic_id] = "Outliers"
                    else:
                        try:
                            topic_words = (
                                topic_clustering.topic_model.get_topic(topic_id)
                            )
                            if topic_words:
                                top_words = [
                                    word for word, _ in topic_words[:3]
                                ]
                                topic_labels[topic_id] = (
                                    f"Topic {topic_id}: {', '.join(top_words)}"
                                )
                            else:
                                topic_labels[topic_id] = f"Topic {topic_id}"
                        except:
                            topic_labels[topic_id] = f"Topic {topic_id}"

                doc_info["topic_label"] = [
                    topic_labels.get(tid, f"Topic {tid}")
                    for tid in doc_info["topic"]
                ]

            # Display sample of documents with their assigned topics
            st.dataframe(
                doc_info[["topic", "topic_label", "document"]].head(50),
                column_config={
                    "document": st.column_config.TextColumn(
                        "Document Content",
                        width="large",
                        help="First 100 characters of document content",
                        max_chars=100,
                    ),
                    "topic": st.column_config.NumberColumn(
                        "Topic ID", help="Topic ID assigned to document"
                    ),
                    "topic_label": st.column_config.TextColumn(
                        "Topic Label", help="Human-readable topic description"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Filter by topic
            if len(doc_info) > 0:
                st.subheader("Filter by Topic")
                topics = sorted(doc_info["topic"].unique())
                selected_topic = st.selectbox(
                    "Select a topic to view its documents:",
                    topics,
                    format_func=lambda x: (
                        "Outliers" if x == -1 else f"Topic {x}"
                    ),
                )

                filtered_docs = doc_info[doc_info["topic"] == selected_topic]

                # Get keywords for this topic if available
                topic_keywords = "N/A"
                if selected_topic != -1:
                    try:
                        words = topic_clustering.topic_model.get_topic(
                            selected_topic
                        )
                        if words:
                            topic_keywords = ", ".join(
                                [word for word, _ in words[:10]]
                            )
                    except:
                        pass

                st.markdown(f"**Keywords:** {topic_keywords}")
                st.markdown(
                    f"**Documents in this topic:** {len(filtered_docs)}"
                )

                # Display documents in this topic
                for i, (_, row) in enumerate(filtered_docs.head(10).iterrows()):
                    with st.expander(f"Document {i+1}"):
                        st.markdown(
                            f"**Probability:** {row.get('probability', 'N/A'):.3f}"
                        )
                        if "title" in row:
                            st.markdown(f"**Title:** {row['title']}")
                        st.text_area(
                            "Content:",
                            row["document"][:1000]
                            + ("..." if len(row["document"]) > 1000 else ""),
                            height=150,
                            key=f"doc_{i}",
                            disabled=True,
                        )
