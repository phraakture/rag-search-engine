import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH / 'movies.json'
STOPWORDS_PATH = DATA_PATH / 'stopwords.txt'

CACHE_PATH = PROJECT_ROOT / 'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data['movies']

_stopwords: list[str] | None = None

def load_stopwords():
    global _stopwords
    if _stopwords is None:
        with open(STOPWORDS_PATH, "r") as f:
            _stopwords = [line.strip() for line in f if line.strip()]
    return _stopwords
