"""Module 5 - Orchestration: query -> classify -> retrieve -> generate -> verify."""
from typing import Dict

from .classifier import RuleClassifier, detect_language, load_category_map
from .config import INDEX_DIR, MIN_CLASSIFIER_CONFIDENCE, TOP_K
from .generator import generate
from .ingestion import build_chunks, save_chunks
from .retriever import HybridRetriever
from .schemas import Answer
from .vector_store import VectorStore


def build_index() -> VectorStore:
    chunks = build_chunks()
    save_chunks(chunks)
    store = VectorStore.build(chunks)
    store.save()
    return store


class LegalRAG:
    def __init__(self, store: VectorStore = None):
        if store is None:
            path = INDEX_DIR / "store.pkl"
            store = VectorStore.load(path) if path.exists() else build_index()
        self.cmap = load_category_map()
        self.classifier = RuleClassifier(self.cmap)
        self.retriever = HybridRetriever(store)

    def ask(self, query: str, k: int = TOP_K) -> Answer:
        query = (query or "").strip()
        lang = detect_language(query)
        cls = self.classifier.classify(query)
        hints = []
        cat = self.cmap["categories"].get(cls.category)
        if cat and cls.confidence >= MIN_CLASSIFIER_CONFIDENCE:
            hints = [tuple(x) for x in cat["laws"]] + [("Constitution of Bangladesh", a) for a in cat["articles"]]
        hits = self.retriever.search(query, hints=hints, k=k)
        return generate(query, lang, cls, hits, self.cmap)
