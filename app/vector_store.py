"""Module 2c - Tiny persistent vector store (swap for Qdrant/pgvector in production, see README)."""
import pickle
from pathlib import Path
from typing import List

import numpy as np

from .config import INDEX_DIR, EMBEDDING_BACKEND, SBERT_MODEL
from .embeddings import get_embedder
from .schemas import Chunk


def chunk_to_text(c: Chunk) -> str:
    return f"{c.law} {'Article' if c.doc_type == 'constitution' else 'Section'} {c.section_no} {c.title}. {c.text}"


class VectorStore:
    def __init__(self, chunks: List[Chunk], embedder, matrix):
        self.chunks, self.embedder, self.matrix = chunks, embedder, matrix

    @classmethod
    def build(cls, chunks: List[Chunk], backend: str = EMBEDDING_BACKEND):
        embedder = get_embedder(backend, SBERT_MODEL)
        matrix = embedder.fit_transform([chunk_to_text(c) for c in chunks])
        return cls(chunks, embedder, matrix)

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.transform([query])
        s = self.matrix @ q.T
        return np.asarray(s.todense()).ravel() if hasattr(s, "todense") else np.asarray(s).ravel()

    def save(self, path: Path = None):
        path = path or (INDEX_DIR / "store.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks, "matrix": self.matrix, "backend": self.embedder.name}
        if self.embedder.name == "tfidf":
            payload["vec"] = self.embedder.vec
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path = None):
        path = path or (INDEX_DIR / "store.pkl")
        with open(path, "rb") as f:
            p = pickle.load(f)
        if p["backend"] == "tfidf":
            emb = get_embedder("tfidf")
            emb.vec = p["vec"]
        else:
            emb = get_embedder("sbert", SBERT_MODEL)
        return cls(p["chunks"], emb, p["matrix"])
