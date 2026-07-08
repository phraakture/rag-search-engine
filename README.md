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

# Hybrid weighted search
uv run cli/hybrid_search_cli.py weighted_search "query" --alpha 0.5 --limit 5

# Hybrid RRF search
uv run cli/hybrid_search_cli.py rrf_search "query" -k 60 --limit 5

# Normalize scores
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

## Pipeline

1. Semantic chunking (sentence-based, configurable size/overlap)
2. Dense chunk embeddings via `all-MiniLM-L6-v2`
3. Hybrid retrieval: BM25 + semantic search with RRF
4. Cross-encoder reranking (planned)
5. Eval + agentic generation (planned)

## Cache

Generated embeddings and indices are stored in `cache/` and ignored by git.
