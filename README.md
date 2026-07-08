# RAG Search Engine

A minimal RAG-style search pipeline over a movie corpus.

## Commands

```bash
# Chunking
uv run cli/semantic_search_cli.py semantic_chunk "text to chunk"

# Build chunk embeddings
uv run cli/semantic_search_cli.py embed_chunks

# Chunked semantic search
uv run cli/semantic_search_cli.py search_chunked "query" --limit 5

# Keyword search
uv run cli/keyword_search_cli.py bm25search "query" --limit 5
```

## Pipeline

1. Semantic chunking (sentence-based, configurable size/overlap)
2. Dense chunk embeddings via `all-MiniLM-L6-v2`
3. Hybrid retrieval: BM25 + semantic search with RRF
4. Cross-encoder reranking (planned)
5. Eval + agentic generation (planned)

## Cache

Generated embeddings and indices are stored in `cache/` and ignored by git.
