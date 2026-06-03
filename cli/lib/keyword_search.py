import pickle
import string
from lib.search_utils import CACHE_PATH, load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import os

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")
        

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text):
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        if tok and tok not in stopwords:
            return True
        return False

    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)

    return res

def has_matching_token(query_tokens, movie_tokens):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False

def search_command(query, n_results=5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    res = []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        matching_doc_ids = idx.get_documents(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res
