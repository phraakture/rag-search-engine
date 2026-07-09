import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


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
    if method != "individual":
        raise ValueError(f"Unknown re-rank method: {method}")

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
