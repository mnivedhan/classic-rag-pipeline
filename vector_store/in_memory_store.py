import chromadb
class InMemoryStore:
    def __init__(self):
        self.client = client = chromadb.PersistentClient(path="./my_chroma_db")
        self.collection = client.get_or_create_collection(name="documents")

    @staticmethod
    def _process_doc(document):
        return {"ids": [document["id"]], "documents": [document["text"]], "metadatas": [document["metadata"]], "embeddings": [document["embedding"]]}

    def insert(self, document):
        processed_doc = self._process_doc(document)
        self.collection.add(**processed_doc)

    def bulk_insert(self, documents):
        for doc in documents:
            self.insert(doc)


    def query(self, query_embeddings, limit=2):
        results = self.collection.query(query_embeddings=query_embeddings, n_results=limit)
        return results