import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


def _get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return api_key


def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_api_key(),
    )


def _call_llm(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content.strip().strip('"') if content else ""


def enhance_query(query: str, method: str) -> str:
    if method == "spell":
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        return _call_llm(prompt) or query

    if method == "rewrite":
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        return _call_llm(prompt) or query

    if method == "expand":
        prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
        expanded = _call_llm(prompt)
        return f"{query} {expanded}" if expanded else query

    raise ValueError(f"Unknown enhancement method: {method}")


def rerank_results(
    query: str, results: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    if method == "individual":
        return _rerank_individual(query, results)
    if method == "batch":
        return _rerank_batch(query, results)
    if method == "cross_encoder":
        return _rerank_cross_encoder(query, results)
    raise ValueError(f"Unknown re-rank method: {method}")


def _rerank_individual(
    query: str, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for result in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {result.get("title", "")} - {result.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        response = _call_llm(prompt)
        try:
            score = float(response.split()[0])
        except ValueError, IndexError:
            score = 0.0
        result["rerank_score"] = max(0.0, min(10.0, score))
        time.sleep(3)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def _rerank_batch(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_lines = [
        f"{result.get('id', 0)}: {result.get('title', '')} - {result.get('document', '')}"
        for result in results
    ]
    doc_list_str = "\n".join(doc_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = _call_llm(prompt)
    try:
        ranked_ids = json.loads(response)
    except json.JSONDecodeError:
        ranked_ids = []

    result_by_id = {result.get("id", 0): result for result in results}
    ranked_results = []
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in result_by_id:
            result_by_id[doc_id]["rerank_rank"] = rank
            ranked_results.append(result_by_id[doc_id])

    for result in results:
        if "rerank_rank" not in result:
            result["rerank_rank"] = len(ranked_results) + 1
            ranked_results.append(result)

    return ranked_results


def _rerank_cross_encoder(
    query: str, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not CROSS_ENCODER_AVAILABLE:
        print(
            "Warning: sentence-transformers not installed. "
            "Falling back to original order."
        )
        return results

    pairs = [
        [query, f"{result.get('title', '')} - {result.get('document', '')}"]
        for result in results
    ]

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    scores = model.predict(pairs, show_progress_bar=False, batch_size=32)

    for result, score in zip(results, scores):
        result["cross_encoder_score"] = float(score)

    return sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)


def evaluate_results(query: str, results: list[dict[str, Any]]) -> list[int | None]:
    formatted_results = [
        f"{r.get('title', '')} - {r.get('document', '')}" for r in results
    ]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = _call_llm(prompt)
    try:
        scores = json.loads(response)
    except json.JSONDecodeError, TypeError:
        return [None] * len(results)

    if not isinstance(scores, list) or len(scores) != len(results):
        return [None] * len(results)

    valid_scores: list[int | None] = []
    for s in scores:
        if isinstance(s, int) and 0 <= s <= 3:
            valid_scores.append(s)
        else:
            valid_scores.append(None)
    return valid_scores
