import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT, load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normalize scores to 0-1")
    norm_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores to normalize"
    )

    ws_parser = subparsers.add_parser(
        "weighted_search", help="Hybrid search with weighted BM25 + semantic scores"
    )
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Weight for BM25 (0-1)"
    )
    ws_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results"
    )

    rrf_parser = subparsers.add_parser(
        "rrf_search", help="Hybrid search with reciprocal rank fusion"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF ranking constant")
    rrf_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "weighted_search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['document']}...")
        case "rrf_search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                bm25_rank = (
                    result["bm25_rank"] if result["bm25_rank"] is not None else "-"
                )
                semantic_rank = (
                    result["semantic_rank"]
                    if result["semantic_rank"] is not None
                    else "-"
                )
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"  {result['document']}...")
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
