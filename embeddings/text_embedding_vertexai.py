import json
from typing import List

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput


class VertexAITextEmbeddings:

    def __init__(self, model = None):
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    def generate_embeddings(self, documents: List[dict]):
        inputs = [TextEmbeddingInput(doc["text"], "RETRIEVAL_DOCUMENT") for doc in documents]
        embeddings = self.model.get_embeddings(inputs)

        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding.values

        return documents

    def generate_embedding(self, query):
        input = [TextEmbeddingInput(query, "RETRIEVAL_DOCUMENT")]
        embedding = self.model.get_embeddings(input)
        return embedding[0].values


    @staticmethod
    def write_to_file(documents, output_file ="output.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
