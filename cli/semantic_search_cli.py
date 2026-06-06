#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)
from lib.semantic_search import (
    chunk_text,
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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document")
    chunk_parser.add_argument("text", type=str, help="Document to be Chunked")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of words in each fixed size chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of overlapping words between chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            for i, result in enumerate(search_command(args.query, args.limit), start=1):
                print(
                    f"{i}. {result['title']} (score: {result['score']:.4f})\n"
                    f"  {result['description']}\n"
                )

        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "verify":
            verify_model()
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
