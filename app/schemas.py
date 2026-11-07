from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Chunk:
    id: str
    law: str                 # e.g. "Penal Code 1860"
    section_no: str          # "379", "32", or "General"
    title: str
    text: str
    doc_type: str = "statute"   # constitution | statute | procedure
    source: str = "seed"        # "seed" or the raw filename
    official_text: bool = False  # False = short paraphrase from the seed index; True = text from official file

    @property
    def label(self) -> str:
        kind = "Article" if self.doc_type == "constitution" else "Section"
        if self.section_no.lower() == "general":
            return f"{self.law}"
        return f"{self.law}, {kind} {self.section_no}"


@dataclass
class Classification:
    category: str
    label: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    emergency: bool = False


@dataclass
class Citation:
    label: str
    law: str
    section_no: str
    title: str
    snippet: str
    official_text: bool
    score: float


@dataclass
class Answer:
    query: str
    language: str
    category: str
    category_label: str
    confidence: float
    emergency: bool
    explanation: str
    constitutional_articles: List[Citation]
    applicable_laws: List[Citation]
    steps: List[str]
    helplines: List[dict]
    warnings: List[str]
    disclaimer: str

    def to_dict(self):
        return asdict(self)
