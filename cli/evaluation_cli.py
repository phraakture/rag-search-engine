import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import GOLDEN_DATASET_PATH, RRF_K, load_movies
from lib.search_utils import GoldenDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET_PATH) as f:
        dataset: GoldenDataset = json.load(f)

    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for case in dataset["test_cases"]:
        query = case["query"]
        relevant = case["relevant_docs"]

        results = hs.rrf_search(query, RRF_K, limit)
        retrieved = [r["title"] for r in results[:limit]]

        relevant_retrieved = sum(1 for t in retrieved if t in relevant)
        precision = relevant_retrieved / limit if limit > 0 else 0.0
        recall = relevant_retrieved / len(relevant) if relevant else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()


if __name__ == "__main__":
    main()
