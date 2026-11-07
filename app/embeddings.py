"""Module 2b - Embedding backends.

tfidf : character n-grams (2-4). Needs no download and works for Bangla and English. Good for the MVP.
sbert : multilingual dense model (default BAAI/bge-m3). Better semantic matching; needs `pip install sentence-transformers`.
"""
from typing import List

import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)

    def fit_transform(self, texts: List[str]):
        return self.vec.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vec.transform(texts)


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]):
        return self.transform(texts)

    def transform(self, texts: List[str]):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def get_embedder(backend: str, sbert_model: str = "BAAI/bge-m3"):
    if backend == "sbert":
        return SbertEmbedder(sbert_model)
    return TfidfEmbedder()
