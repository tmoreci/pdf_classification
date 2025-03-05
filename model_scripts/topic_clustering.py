from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import os
from dotenv import load_dotenv
from db import DocumentDatabase


class TopicClustering:
    """
    A class for performing different types of topic modeling on document collections.

    This class encapsulates various topic modeling approaches using BERTopic:
    - Basic unsupervised topic modeling
    - Guided topic modeling with seed topics
    - Zero-shot topic modeling with predefined topics

    It integrates with the DocumentDatabase class to access document collections.
    """

    def __init__(self, database=None, embedding_model="BAAI/bge-base-en-v1.5"):
        """
        Initialize the TopicClustering instance.

        Args:
            database: DocumentDatabase instance for accessing documents
            embedding_model: Name or instance of the sentence transformer embedding model
        """
        self.database = database
        self.documents = []
        self.document_embeddings = []
        self.document_metadata = []
        self.topic_model = None
        self.topics = None
        self.probs = None

        # Initialize embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

    def load_documents_from_db(self, filter_criteria=None):
        """
        Load documents and their embeddings from the connected database.
        """
        if not self.database:
            raise ValueError("No database connection available")

        # Retrieve all documents from the ChromaDB collection with embeddings
        collection_info = self.database.collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        self.documents = collection_info.get("documents", [])
        self.document_metadata = collection_info.get("metadatas", [])
        self.document_embeddings = collection_info.get("embeddings", [])

        # Apply filtering if criteria provided
        if filter_criteria and self.document_metadata:
            filtered_indices = []
            for i, metadata in enumerate(self.document_metadata):
                if all(
                    metadata.get(k) == v for k, v in filter_criteria.items()
                ):
                    filtered_indices.append(i)

            self.documents = [self.documents[i] for i in filtered_indices]
            self.document_metadata = [
                self.document_metadata[i] for i in filtered_indices
            ]
            if self.document_embeddings:
                self.document_embeddings = [
                    self.document_embeddings[i] for i in filtered_indices
                ]

        return self.documents

    def basic_topic_modeling(
        self,
        min_topic_size=5,
        nr_topics="auto",
        verbose=True,
    ):
        """
        Perform basic unsupervised topic modeling using precomputed embeddings.
        """
        if not self.documents:
            raise ValueError(
                "No documents loaded. Please load documents first."
            )

        # Configure BERTopic model without embedding model if using precomputed embeddings
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            representation_model=KeyBERTInspired(),
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
        )
        self.topics, self.probs = self.topic_model.fit_transform(self.documents)

        if verbose:
            topic_info = self.topic_model.get_topic_info()
            print(f"Found {len(topic_info) - 1} topics (excluding outliers)")
            print(topic_info)

        return self.topics, self.probs

    def zero_shot_topic_modeling(
        self, topic_list, min_similarity=0.4, min_topic_size=3, verbose=True
    ):
        """
        Perform zero-shot topic modeling with predefined topics.

        Args:
            topic_list: List of topic names
            min_similarity: Minimum similarity threshold for assigning documents to topics
            min_topic_size: Minimum size of topics
            verbose: Whether to print topic information

        Returns:
            List of assigned topics
        """
        if not self.documents:
            raise ValueError(
                "No documents loaded. Please load documents first."
            )

        # Configure and fit BERTopic model with zero-shot topics
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=min_topic_size,
            zeroshot_topic_list=topic_list,
            zeroshot_min_similarity=min_similarity,
            representation_model=KeyBERTInspired(),
        )
        # ! ToDO align embedding models so I can add precomputed embeddings here
        self.topics, self.probs = self.topic_model.fit_transform(self.documents)

        if verbose:
            topic_info = self.topic_model.get_topic_info()
            print(f"Found {len(topic_info) - 1} topics (excluding outliers)")
            print(topic_info)

        return self.topics, self.probs

    def visualize_topics(self, width=800, height=600):
        """
        Visualize the discovered topics with fallback to basic bar chart if UMAP visualization fails.

        The primary visualization uses BERTopic's built-in visualization which requires sufficient topics.
        If that fails (typically due to too few topics), a basic bar chart showing document counts per topic
        is displayed instead.

        Args:
            width: Width of the visualization
            height: Height of the visualization

        Returns:
            Figure object from the visualization
        """
        if not self.topic_model:
            raise ValueError(
                "No topic model available. Run a topic modeling method first."
            )

        try:
            # First attempt to use BERTopic's built-in visualization
            fig = self.topic_model.visualize_topics(width=width, height=height)
            return fig
        except (ValueError, Exception) as e:
            # Fallback to basic bar chart visualization
            print(f"BERTopic visualization failed: {str(e)}")
            print("Falling back to basic topic count visualization")

            # Count documents per topic
            topic_counts = Counter(self.topics)

            # Sort topics by their IDs
            sorted_topics = sorted(topic_counts.keys())

            # Create figure with appropriate size (convert pixels to inches for matplotlib)
            fig_width = width / 100  # Rough approximation for inch conversion
            fig_height = height / 100
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Create labels with topic keywords when available
            x_labels = []
            for topic_id in sorted_topics:
                if topic_id == -1:
                    x_labels.append("Outliers")
                else:
                    # Try to get topic keywords if available
                    try:
                        topic_info = self.topic_model.get_topic(topic_id)
                        if topic_info and len(topic_info) > 0:
                            # Use the top keyword to identify the topic
                            keyword = topic_info[0][0]
                            x_labels.append(f"Topic {topic_id}: {keyword}")
                        else:
                            x_labels.append(f"Topic {topic_id}")
                    except:
                        x_labels.append(f"Topic {topic_id}")

            # Set colors (red for outliers, blue for regular topics)
            colors = [
                "#ff9999" if topic_id == -1 else "#5599ff"
                for topic_id in sorted_topics
            ]

            # Create bar chart
            bars = ax.bar(
                x_labels,
                [topic_counts[topic_id] for topic_id in sorted_topics],
                color=colors,
            )

            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )

            # Set title and labels
            ax.set_title("Document Count by Topic")
            ax.set_xlabel("Topics")
            ax.set_ylabel("Number of Documents")

            # Rotate labels if there are many topics
            if len(x_labels) > 5:
                plt.xticks(rotation=45, ha="right")

            plt.tight_layout()

            return fig

    def visualize_document_distribution(
        self, figsize=(10, 6), top_n=10, show_counts=True
    ):
        """
        Visualize the distribution of documents across topics with a horizontal bar chart.

        Args:
            figsize: Tuple of (width, height) for the figure size
            top_n: Number of top topics to display (excluding outliers)
            show_counts: Whether to show the document count next to each bar

        Returns:
            Matplotlib figure object
        """
        if not self.topic_model or self.topics is None:
            raise ValueError(
                "No topic model available. Run a topic modeling method first."
            )

        # Count documents per topic
        topic_counts = Counter(self.topics)

        # Get topic information with labels
        topic_info = self.topic_model.get_topic_info()

        # Create a mapping from topic_id to representative words
        topic_labels = {}
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                topic_labels[topic_id] = "Outliers"
            else:
                # Get the words for this topic
                words = row["Name"].split("_")
                if len(words) > 1:  # Ensure we have some words
                    label = f"Topic {topic_id}: {', '.join(words[:3])}"
                    topic_labels[topic_id] = label
                else:
                    topic_labels[topic_id] = f"Topic {topic_id}"

        # Filter to top N topics + outliers
        topic_data = [(tid, count) for tid, count in topic_counts.items()]
        topic_data.sort(
            key=lambda x: (x[0] != -1, -x[1])
        )  # Sort, keeping outliers first if present

        # Limit to top N (but always include outliers if present)
        if -1 in topic_counts:
            if len(topic_data) > top_n + 1:  # +1 for outliers
                topic_data = topic_data[: top_n + 1]
        else:
            if len(topic_data) > top_n:
                topic_data = topic_data[:top_n]

        # Extract data for plotting
        topics = [
            topic_labels.get(tid, f"Topic {tid}") for tid, _ in topic_data
        ]
        counts = [count for _, count in topic_data]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)

        # Define colors (red for outliers, blue gradient for other topics)
        colors = []
        for tid, _ in topic_data:
            if tid == -1:
                colors.append("#ff9999")  # Red for outliers
            else:
                # Generate a blue gradient based on position
                position = list(topic_data).index((tid, topic_counts[tid]))
                if position > 0:  # Skip outliers in position calculation
                    position -= 1 if -1 in topic_counts else 0
                intensity = 0.5 + (
                    0.5 * (position / max(1, len(topic_data) - 2))
                )
                # Fix: Ensure RGB values don't exceed 255 (FF in hex)
                colors.append(
                    f"#{min(255, int(intensity * 60)):02x}{min(255, int(intensity * 90)):02x}{min(255, int(intensity * 255)):02x}"
                )

        # Create horizontal bars
        bars = ax.barh(topics, counts, color=colors)

        # Add count labels
        if show_counts:
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width + (max(counts) * 0.01),  # Slight offset
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    va="center",
                )

        # Set title and labels
        ax.set_title("Document Distribution Across Topics")
        ax.set_xlabel("Number of Documents")

        # Adjust layout
        plt.tight_layout()

        return fig

    def visualize_zero_shot_topics(self, figsize=(10, 6), save_path=None):
        """
        Visualize the results of zero-shot topic modeling using a simple bar chart.

        Args:
            figsize: Tuple of (width, height) for the figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        if not self.topic_model:
            raise ValueError(
                "No topic model available. Run a topic modeling method first."
            )

        # Count documents per topic
        topic_counts = Counter(self.topics)

        # Sort topics and prepare topic labels
        sorted_topics = sorted(topic_counts.keys())

        # Use zero-shot topic names if available
        topic_labels = {}
        if (
            hasattr(self.topic_model, "zeroshot_topic_list")
            and self.topic_model.zeroshot_topic_list
        ):
            for i, name in enumerate(self.topic_model.zeroshot_topic_list):
                topic_labels[i] = name

        # Create labels for x-axis
        x_labels = []
        for topic_id in sorted_topics:
            if topic_id == -1:
                x_labels.append("Outliers")
            else:
                topic_name = topic_labels.get(topic_id, f"Topic {topic_id}")
                x_labels.append(topic_name)

        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize)

        # Set colors (red for outliers, blue for regular topics)
        colors = [
            "#ff9999" if label == "Outliers" else "#5599ff"
            for label in x_labels
        ]

        # Create bar chart
        bars = ax.bar(
            x_labels,
            [topic_counts[topic_id] for topic_id in sorted_topics],
            color=colors,
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

        # Set title and labels
        ax.set_title("Document Count by Zero-Shot Topic")
        ax.set_xlabel("Topics")
        ax.set_ylabel("Number of Documents")

        # Rotate labels if there are many topics
        if len(x_labels) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_topic_info(self):
        """
        Get information about the discovered topics.

        Returns:
            DataFrame with topic information
        """
        if not self.topic_model:
            raise ValueError(
                "No topic model available. Run a topic modeling method first."
            )

        return self.topic_model.get_topic_info()

    def get_document_info(self):
        """
        Get document-topic assignments with metadata.

        Returns:
            DataFrame with document topics and metadata
        """
        if not self.topic_model or self.topics is None:
            raise ValueError(
                "No topic model available. Run a topic modeling method first."
            )

        # Create DataFrame with document topics
        doc_info = pd.DataFrame(
            {
                "document": self.documents,
                "topic": self.topics,
                "probability": self.probs,
            }
        )

        # Add metadata if available
        if self.document_metadata:
            for key in self.document_metadata[0].keys():
                doc_info[key] = [
                    meta.get(key) for meta in self.document_metadata
                ]

        return doc_info

    def save_model(self, path):
        """
        Save the topic model to disk.

        Args:
            path: Directory path to save the model

        Returns:
            Path where model was saved
        """
        if not self.topic_model:
            raise ValueError("No topic model available to save")

        self.topic_model.save(path)
        return path

    def load_model(self, path):
        """
        Load a topic model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded topic model
        """
        self.topic_model = BERTopic.load(path)
        return self.topic_model


# Example usage
if __name__ == "__main__":

    load_dotenv(".env")

    # Initialize database and topic clustering
    database = DocumentDatabase(gemini_api=os.getenv("GOOGLE_API"))
    topic_clustering = TopicClustering(database)

    # Load documents
    documents = topic_clustering.load_documents_from_db()

    # topics, probs = topic_clustering.zero_shot_topic_modeling(
    #     ["Agents", "Retrieval", "Training"]
    # )
    topics, probs = topic_clustering.basic_topic_modeling()
    print(f"Found topics: {np.unique(topics)}")
    df = topic_clustering.get_document_info().head()
    print(df)
    print("Trying visualization")
    # Try visualizing topics with fallback mechanism
    fig = topic_clustering.visualize_document_distribution()
    if fig:
        plt.show()
