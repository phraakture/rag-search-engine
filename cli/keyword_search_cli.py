#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")
    build_parser.set_defaults(func=lambda: build_command())

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to find frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Search term to find idf for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF for a document and term"
    )
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to find frequency for")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query, 5)
            print(f"Searching for: {args.query}")
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
