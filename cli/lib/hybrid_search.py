import os
from collections.abc import Mapping, Sequence
from typing import Any

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DOCUMENT_PREVIEW_LENGTH


class HybridSearch:
    def __init__(self, documents: Sequence[Mapping[str, Any]]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)  # type: ignore[return-value]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search(query, search_limit)

        bm25_scores = [r["score"] for r in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]

        norm_bm25_scores = normalize_scores(bm25_scores)
        norm_semantic_scores = normalize_scores(semantic_scores)

        docs_by_id: dict[int, dict] = {}

        for result, norm_score in zip(bm25_results, norm_bm25_scores):
            doc_id = result["id"]
            docs_by_id[doc_id] = {
                "doc": result,
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        title_to_doc = {doc["title"]: doc for doc in self.documents}
        for result, norm_score in zip(semantic_results, norm_semantic_scores):
            title = result["title"]
            doc = title_to_doc.get(title)
            if doc is None:
                continue
            doc_id = doc["id"]
            if doc_id in docs_by_id:
                docs_by_id[doc_id]["semantic_score"] = norm_score
            else:
                docs_by_id[doc_id] = {
                    "doc": doc,
                    "bm25_score": 0.0,
                    "semantic_score": norm_score,
                }

        for info in docs_by_id.values():
            info["hybrid_score"] = hybrid_score(
                info["bm25_score"], info["semantic_score"], alpha
            )

        ranked = sorted(
            docs_by_id.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        results = []
        for info in ranked[:limit]:
            doc = info["doc"]
            results.append(
                {
                    "id": doc.get("id", 0),
                    "title": doc["title"],
                    "document": doc.get("description", doc.get("document", ""))[
                        :DOCUMENT_PREVIEW_LENGTH
                    ],
                    "bm25_score": info["bm25_score"],
                    "semantic_score": info["semantic_score"],
                    "hybrid_score": info["hybrid_score"],
                }
            )

        return results

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search(query, search_limit)

        docs_by_id: dict[int, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            docs_by_id[doc_id] = {
                "doc": result,
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        title_to_doc = {doc["title"]: doc for doc in self.documents}
        for rank, result in enumerate(semantic_results, start=1):
            title = result["title"]
            doc = title_to_doc.get(title)
            if doc is None:
                continue
            doc_id = doc["id"]
            if doc_id in docs_by_id:
                docs_by_id[doc_id]["semantic_rank"] = rank
                docs_by_id[doc_id]["rrf_score"] += rrf_score(rank, k)
            else:
                docs_by_id[doc_id] = {
                    "doc": doc,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }

        ranked = sorted(docs_by_id.values(), key=lambda x: x["rrf_score"], reverse=True)

        results = []
        for info in ranked[:limit]:
            doc = info["doc"]
            results.append(
                {
                    "id": doc.get("id", 0),
                    "title": doc["title"],
                    "document": doc.get("description", doc.get("document", ""))[
                        :DOCUMENT_PREVIEW_LENGTH
                    ],
                    "rrf_score": info["rrf_score"],
                    "bm25_rank": info["bm25_rank"],
                    "semantic_rank": info["semantic_rank"],
                }
            )

        return results


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
