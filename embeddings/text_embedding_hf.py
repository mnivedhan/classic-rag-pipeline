import json
from typing import List

from sentence_transformers import SentenceTransformer

class HuggingFaceTextEmbeddings:
    def __init__(self, model = None):
        self.model = model or SentenceTransformer("BAAI/bge-base-en-v1.5")

    def generate_embeddings(self, documents: List[dict]):
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        for doc, embedding in zip(documents, embeddings):
            # Truncate rounding to 4 decimals (optional, matching your format)
            doc["embedding"] = [round(float(x), 4) for x in embedding]

        return documents

    def generate_embedding(self, query):
        embedding = self.model.encode(query, convert_to_numpy=True)
        return [round(float(x), 4) for x in embedding]


    @staticmethod
    def write_to_file(documents, output_file ="output.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
