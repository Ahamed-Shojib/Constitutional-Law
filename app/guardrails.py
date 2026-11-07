"""Module 4b - Guardrails: never let the answer cite a provision that was not retrieved."""
import re
from typing import List, Tuple

from .ingestion import normalize_digits
from .schemas import Chunk

CITE_RE = re.compile(r"(?:Section|Sec\.?|Article|ধারা|অনুচ্ছেদ)\s*(\d{1,3}[A-Z]?)|(\d{1,3}[A-Z]?)\s*(?:ধারা|অনুচ্ছেদ)", re.I)
SENT_SPLIT = re.compile(r"(?<=[\.!?।])\s+")


def cited_numbers(text: str) -> List[str]:
    return [a or b for a, b in CITE_RE.findall(normalize_digits(text))]


def verify_explanation(text: str, hits: List[Chunk], extra_allowed: List[str] = None) -> Tuple[str, List[str]]:
    """Drop sentences that cite a section/article number that is not in the retrieved context."""
    allowed = {c.section_no.upper() for c in hits} | {a.upper() for a in (extra_allowed or [])}
    kept, removed = [], []
    for sent in SENT_SPLIT.split(text.strip()):
        bad = [n for n in cited_numbers(sent) if n.upper() not in allowed]
        (removed if bad else kept).append(sent)
    warnings = [f"Removed an unverified citation from the AI explanation: '{s[:80]}...'" for s in removed]
    return " ".join(kept).strip(), warnings
