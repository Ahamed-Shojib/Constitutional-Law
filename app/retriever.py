"""Module 3b - Hybrid retriever: BM25 (exact words / section numbers) + dense/TF-IDF vectors,
merged with Reciprocal Rank Fusion (RRF), then boosted by the classifier's category hints.
A cross-encoder reranker can be plugged in at `rerank()` (see README)."""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ingestion import normalize_digits
from .schemas import Chunk
from .vector_store import VectorStore, chunk_to_text

TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+")  # \u0980-\u09FF keeps Bangla vowel signs inside words


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_digits(text).lower())


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in query_tokens:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _ranks(scores: np.ndarray) -> Dict[int, int]:
    order = np.argsort(-scores)
    return {int(idx): r for r, idx in enumerate(order) if scores[idx] > 0}


class HybridRetriever:
    def __init__(self, store: VectorStore, rrf_k: int = 60, hint_boost: float = 0.05):
        self.store = store
        self.chunks = store.chunks
        self.bm25 = BM25([tokenize(chunk_to_text(c)) for c in self.chunks])
        self.rrf_k, self.hint_boost = rrf_k, hint_boost
        self.index_of = {(c.law, c.section_no): i for i, c in enumerate(self.chunks)}

    def search(self, query: str, hints: Optional[Sequence[Tuple[str, str]]] = None, k: int = 6, max_articles: int = 3) -> List[Tuple[Chunk, float]]:
        bm = self.bm25.scores(tokenize(query))
        dn = self.store.scores(query)
        fused: Dict[int, float] = {}
        for ranks in (_ranks(bm), _ranks(dn)):
            for idx, r in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + r + 1)
        # classifier hints: pull the category's known provisions into the candidate pool and boost them
        for law, sec in hints or []:
            idx = self.index_of.get((law, sec))
            if idx is not None:
                fused[idx] = fused.get(idx, 0.0) + self.hint_boost
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        # separate slots so constitutional articles never crowd out the statutes (k statutes + up to max_articles articles)
        laws = [(self.chunks[i], s) for i, s in ranked if self.chunks[i].doc_type != "constitution"][:k]
        arts = [(self.chunks[i], s) for i, s in ranked if self.chunks[i].doc_type == "constitution"][:max_articles]
        merged = sorted(laws + arts, key=lambda x: x[1], reverse=True)
        return self.rerank(query, merged)

    def rerank(self, query: str, candidates: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Hook: plug in a cross-encoder (e.g. BAAI/bge-reranker-v2-m3) here. Default keeps RRF order."""
        return candidates
