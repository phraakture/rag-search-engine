import os

from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    MOVIE_EMBEDDINGS_PATH,
    DEFAULT_CHUNK_SIZE,
    load_movies,
    DEFAULT_CHUNK_OVERLAP,
)


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = MOVIE_EMBEDDINGS_PATH

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
            return self.build_embeddings(documents)
        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = np.asarray(self.generate_embedding(query))
        embeddings_array = np.asarray(self.embeddings)

        scored = []
        for doc, doc_embedding in zip(self.documents, embeddings_array):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scored[:limit]
        ]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def fixed_sized_chunking(
    text, overlap=DEFAULT_CHUNK_OVERLAP, chunk_size=DEFAULT_CHUNK_SIZE
):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_sized_chunking(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    ss = SemanticSearch()
    ss.load_or_create_embeddings(load_movies())
    return ss.search(query, limit)
