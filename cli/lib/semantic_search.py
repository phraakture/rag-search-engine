import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DOCUMENT_PREVIEW_LENGTH,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    CACHE_DIR,
    format_search_result,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = CHUNK_EMBEDDINGS_PATH
        self.chunk_metadata_path = CHUNK_METADATA_PATH

    def build_chunk_embeddings(
        self, documents: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for midx, doc in enumerate(documents):
            if not doc["description"].strip():
                continue
            chunks = semantic_chunking(doc["description"], max_chunk_size=4, overlap=1)
            for cidx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": midx, "chunk_idx": cidx, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = np.asarray(self.generate_embedding(query))

        chunk_scores = []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[idx]
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": score,
                }
            )

        movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            midx = chunk_score["movie_idx"]
            if midx not in movie_scores or chunk_score["score"] > movie_scores[midx]:
                movie_scores[midx] = chunk_score["score"]

        sorted_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_scores = sorted_scores[:limit]

        results = []
        for midx, score in top_scores:
            doc = self.documents[midx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results


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


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = 0,
) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []
    step_size = max_chunk_size - overlap
    if step_size <= 0:
        return [" ".join(sentences)]
    chunks: list[str] = []
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = 0,
) -> None:
    chunks = semantic_chunking(text, max_chunk_size=max_chunk_size, overlap=overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


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
