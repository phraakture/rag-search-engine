#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search_command,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="Verify the model loads correctly")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Encode text with embedding model"
    )
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    subparsers.add_parser("verify_embeddings", help="Verify embedding")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            for result in search_command(args.query, args.limit):
                print(
                    f"{result['title']} (score: {result['score']})\n"
                    f"  {result['document']}\n"
                )
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_query_text(args.query)

        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
