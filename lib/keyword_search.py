import json
import string
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIES_PATH = DATA_DIR / "movies.json"

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def search_command(query: str, n_results):
    with open(MOVIES_PATH) as f:
        data = json.load(f)

    results = []

    for movie in data["movies"]:
        if query.lower() in movie["title"].lower():
            results.append(movie)

        if len(results) >= n_results:
            break

    return results
