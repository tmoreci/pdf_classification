from db import DocumentDatabase
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os


def basic_fitting(documents):
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(documents)
    print(topic_model.get_topic_info())
    return topics, probs


def guided_fitting(documents, seed_topics):
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    topic_model = BERTopic(
        embedding_model=embedding_model, seed_topic_list=seed_topics
    )
    topics, probs = topic_model.fit_transform(documents)
    print(topic_model.get_topic_info())
    return topics, probs


def zero_shot(documents, topic_list, min_similarity=0.4):
    topic_model = BERTopic(
        embedding_model="BAAI/bge-base-en-v1.5",
        min_topic_size=3,
        zeroshot_topic_list=topic_list,
        zeroshot_min_similarity=min_similarity,
        representation_model=KeyBERTInspired(),
    )
    topics, _ = topic_model.fit_transform(documents)
    topic_model.visualize_topics()
    return topics


if __name__ == "__main__":
    load_dotenv(".env")

    database = DocumentDatabase(gemini_api=os.getenv("GOOGLE_API"))
    # Retrieve all documents from the ChromaDB collection
    collection_info = database.collection.get()
    all_documents = collection_info.get("documents", [])
    print(all_documents[0])
    seed_topics = [
        ["agents", "systems"],
        ["retrieval", "RAG"],
        ["fine-tuning", "training"],
    ]
    topics = ["Agents", "Retrieval", "Training"]
    # print(guided_fitting(all_documents, seed_topics))
    print(zero_shot(all_documents, topics))
