import argparse

from lib.hybrid_search import HybridSearch
from lib.llm import _call_llm
from lib.search_utils import RRF_K, load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Summarize search results using RAG"
    )
    sum_parser.add_argument("query", type=str, help="Search query to summarize")
    sum_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results"
    )

    cite_parser = subparsers.add_parser(
        "citations", help="Answer with source citations"
    )
    cite_parser.add_argument("query", type=str, help="Search query")
    cite_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, RRF_K, 5)
            titles = [r["title"] for r in results]

            docs_text = "\n".join(
                f"{r['title']}: {r.get('document', '')}" for r in results
            )

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs_text}

Answer:"""

            answer = _call_llm(prompt)

            print("Search Results:")
            for title in titles:
                print(f"- {title}")
            print(f"\nRAG Response:\n{answer}")
        case "summarize":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, RRF_K, args.limit)
            titles = [r["title"] for r in results]

            results_text = "\n".join(
                f"{r['title']}: {r.get('document', '')}" for r in results
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results_text}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            summary = _call_llm(prompt)

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            print(f"\nLLM Summary:\n{summary}")
        case "citations":
            query = args.query

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, RRF_K, args.limit)
            titles = [r["title"] for r in results]

            docs_text = "\n".join(
                f"[{i+1}] {r['title']}: {r.get('document', '')}"
                for i, r in enumerate(results)
            )

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs_text}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            answer = _call_llm(prompt)

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            print(f"\nLLM Answer:\n{answer}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
