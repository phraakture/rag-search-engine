import os

from networkx import reverse

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from cli.lib.hybrid_search import normalize_scores


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        combined_results = combine_search_results(bm25_results, sem_results)
        return combined_results


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def hybrid_score(bm25_score, sem_score, alpha=0.5):
    return (alpha * bm25_score) + ((1-alpha) * sem_score)

def normalize_search_results(results):
    scores = [r["score"] for r in results]
    norm_scores = normalize_scores(scores)
    for idx, result in enumerate(results):
        result["normalized_score"] = norm_scores[idx]
    return results

def combine_search_results(bm25_results, sem_results):
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_scores(sem_results)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm["doc_id"]
        combined_norm[doc_id] = {
                "doc_id": doc_id,
                "bm25_score": norm["normalized_score"],
                "sem_score": 0.,
                "title": norm["title"],
                "description": norm['description']
        }
    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                    "doc_id": doc_id,
                    "bm25_score": 0.,
                    "sem_score": 0.,
                    "title": norm["title"],
                    "description": norm['description']
            }
        combined_norm[doc_id]['sem_score'] = norm["normalized_score"]

    for k,v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v["bm25_score", v["sem_score"]])

    results = sorted(combined_norm.values(), key=lambda x: x[0], reverse = True)
    return results




