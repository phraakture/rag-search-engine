#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    load_movies,
)
from lib.semantic_search import (
    ChunkedSemanticSearch,
    chunk_text,
    embed_query_text,
    embed_text,
    search_command,
    semantic_chunk,
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

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-based chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_SEMANTIC_CHUNK_SIZE,
        help="Maximum sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences between chunks",
    )

    subparsers.add_parser(
        "embed_chunks", help="Build or load chunk embeddings for all documents"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked semantic similarity"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)

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
        case "embed_chunks":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            total_chunks = len(css.chunk_metadata) if css.chunk_metadata else 0
            print(f"Total documents: {len(documents)}")
            print(f"Total chunks: {total_chunks}")
            print(f"Embeddings shape: {embeddings.shape}")
        case "search_chunked":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
