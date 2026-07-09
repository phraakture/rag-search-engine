from typing import Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(
        self, documents: list[dict[str, Any]], model_name: str = "clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        self.text_embeddings = self.model.encode(
            self.texts, show_progress_bar=True, batch_size=64
        )

    def embed_image(self, image_path: str) -> list[float]:
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self, image_path: str, top_k: int = 5) -> list[dict[str, Any]]:
        image_emb = self.embed_image(image_path)

        scores = [
            float(np.dot(image_emb, text_emb))
            for text_emb in self.text_embeddings
        ]

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": round(scores[idx], 3),
            })
        return results


def verify_image_embedding(image_path: str) -> None:
    ms = MultimodalSearch([])
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict[str, Any]]:
    documents = load_movies()
    ms = MultimodalSearch(documents)
    return ms.search_with_image(image_path)
