"""Module 3a - Incident classifier (rule/keyword based, English + Bangla).

Maps a free-text story to a crime/legal category, which then (a) steers retrieval and (b) selects action steps.
Upgrade path: replace `classify()` with an LLM few-shot JSON classifier or a fine-tuned BanglaBERT; keep the same
return type so nothing else changes.
"""
import json
import re
from typing import Dict, List

from .config import SEED_DIR
from .schemas import Classification
from .ingestion import normalize_digits


def load_category_map() -> Dict:
    return json.loads((SEED_DIR / "category_map.json").read_text(encoding="utf-8"))


def detect_language(text: str) -> str:
    bn = len(re.findall(r"[\u0980-\u09FF]", text))
    letters = len(re.findall(r"[A-Za-z\u0980-\u09FF]", text)) or 1
    return "bn" if bn / letters > 0.3 else "en"


class RuleClassifier:
    def __init__(self, cmap: Dict = None):
        self.cmap = cmap or load_category_map()
        self.cats = self.cmap["categories"]
        em = self.cmap["emergency_keywords"]
        self.emergency_kw = [k.lower() for k in em["en"] + em["bn"]]

    def _score(self, text: str, cat: Dict) -> float:
        hits = 0.0
        for kw in cat["kw_en"] + cat["kw_bn"]:
            k = kw.lower()
            if k in text:
                hits += 1.0 + min(len(k), 20) / 20.0  # longer, more specific keywords count more
        return hits

    def classify(self, query: str) -> Classification:
        text = normalize_digits(query).lower()
        scores = {cid: self._score(text, c) for cid, c in self.cats.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top = ranked[0]
        total = sum(scores.values())
        emergency = any(k in text for k in self.emergency_kw)
        if top <= 0:
            return Classification("unknown", "Unclassified", 0.0, [], emergency)
        confidence = round(top / total, 2) if total else 0.0
        alts = [cid for cid, s in ranked[1:4] if s > 0]
        return Classification(top_id, self.cats[top_id]["label"], confidence, alts, emergency)
