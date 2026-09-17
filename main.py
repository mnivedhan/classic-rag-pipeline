from embeddings.text_embedding_hf import HuggingFaceTextEmbeddings
from vector_store.in_memory_store import InMemoryStore

if __name__ == '__main__':

    docs = [
        {
            "id": "doc_001",
            "text": "Redis is an in-memory data store commonly used for caching, session management, and fast key-value lookups.",
            "metadata": {"source": "redis_guide", "category": "database", "language": "en"}
        },
        {
            "id": "doc_002",
            "text": "Kafka is a distributed event streaming platform designed to handle high-throughput, fault-tolerant streams of events.",
            "metadata": {"source": "kafka_guide", "category": "messaging", "language": "en"}
        },
        {
            "id": "doc_003",
            "text": "Vector databases store numerical representations of text, images, and other data so that semantically similar items can be retrieved efficiently.",
            "metadata": {"source": "vector_search_guide", "category": "ai", "language": "en"}
        },
        {
            "id": "doc_004",
            "text": "A task queue allows producers to submit jobs and consumers to process those jobs asynchronously.",
            "metadata": {"source": "distributed_systems", "category": "architecture", "language": "en"}
        },
        {
            "id": "doc_005",
            "text": "RAG combines information retrieval with large language models by retrieving relevant documents and providing them as context to the model.",
            "metadata": {"source": "rag_guide", "category": "genai", "language": "en"}
        }
    ]

    text_embedding = HuggingFaceTextEmbeddings()
    in_memory_store = InMemoryStore()
    doc_with_embeddings = text_embedding.generate_embeddings(docs)
    # text_embedding.write_to_file(doc_embeddings)
    in_memory_store.bulk_insert(doc_with_embeddings)

    search_query = "kafka vs queue"
    query_embedding = text_embedding.generate_embedding(search_query)
    result = in_memory_store.query(query_embedding, limit=10)
    print(result)