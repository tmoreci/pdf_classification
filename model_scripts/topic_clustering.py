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
        print("Using precomputed embeddings from ChromaDB")
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
        self.topics, self.probs = self.topic_model.fit_transform(self.documents)

        if verbose:
            topic_info = self.topic_model.get_topic_info()
            print(f"Found {len(topic_info) - 1} topics (excluding outliers)")
            print(topic_info)

        return self.topics

    def visualize_topics(self, width=800, height=600):
        """
        Visualize the discovered topics.

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
        # ! To DO: write my own visualization function
        if len(self.topics) < 10:
            print("Not enough topics to visualize without UMAP error")
            return None
        fig = self.topic_model.visualize_topics(width=width, height=height)
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
    # print(f"Loaded {len(documents)} documents from the database")

    # # Example 1: Basic topic modeling
    # print("\nPerforming basic topic modeling...")
    topics, probs = topic_clustering.basic_topic_modeling(min_topic_size=2)
    print(f"Found topics: {np.unique(topics)}")
    # print(f"Number of documents: {len(topic_clustering.documents)}")
    # # Example 3: Zero-shot topic modeling
    # print("\nPerforming zero-shot topic modeling...")
    # predefined_topics = ["Agents", "Retrieval", "Training", "Evaluation"]
    # topics = topic_clustering.zero_shot_topic_modeling(predefined_topics)

    # Visualize results
    # Example usage:
    # Example usage:
    # predefined_topics = ["Agents", "Retrieval", "Training", "Evaluation"]
    # topics = topic_clustering.zero_shot_topic_modeling(predefined_topics)
    df = topic_clustering.get_document_info().head()
    print(df)
    topic_clustering.visualize_topics()
    # Create and show visualization
    # fig = topic_clustering.visualize_zero_shot_topics()
    # plt.show()
