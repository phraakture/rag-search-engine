import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIES_PATH = DATA_DIR / "movies.json"


def search_command(query: str, max_results: int = 5) -> list[dict]:
    with open(MOVIES_PATH) as f:
        data = json.load(f)

    results = []
    for movie in data["movies"]:
        if query in movie["title"]:
            results.append(movie)
        if len(results) >= max_results:
            break

    return results
