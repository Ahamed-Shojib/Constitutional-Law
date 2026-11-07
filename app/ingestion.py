"""Module 2a - Ingestion: turn law files into section-level chunks.

Raw file convention (data/raw/*.txt|*.md|*.pdf):

    # LAW: Penal Code 1860 | TYPE: statute        <- type: constitution | statute | procedure
    378. Theft
    Whoever, intending to take dishonestly ...
    379. Punishment for theft
    Whoever commits theft shall be ...

A line that starts with a number (optionally "Section"/"Article"/"ধারা"/"অনুচ্ছেদ") followed by '.' or ')'
starts a new chunk. One chunk = one section/article, which is how lawyers cite law.
"""
import json
import re
from pathlib import Path
from typing import List

from .config import RAW_DIR, SEED_DIR, INDEX_DIR
from .schemas import Chunk

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
HEADER_RE = re.compile(r"^#\s*LAW:\s*(.+?)(?:\s*\|\s*TYPE:\s*(\w+))?\s*$", re.I)
SECTION_RE = re.compile(
    r"^\s*(?:(?:Section|Sec\.?|Article|ধারা|অনুচ্ছেদ)\s*)?(\d{1,3}[A-Z]?)\s*[\.\):–\-]\s*(.*)$"
)
MAX_CHARS = 1500


def normalize_digits(text: str) -> str:
    return text.translate(BN_DIGITS)


def _split_long(text: str) -> List[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=[\.।])\s+", text):
        if len(cur) + len(para) > MAX_CHARS and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_text(text: str, default_law: str, default_type: str, source: str) -> List[Chunk]:
    law, doc_type = default_law, default_type
    chunks: List[Chunk] = []
    cur = None  # (section_no, title, [lines])

    def flush():
        nonlocal cur
        if not cur:
            return
        sec, title, lines = cur
        body = " ".join(l.strip() for l in lines if l.strip()).strip() or title
        for i, part in enumerate(_split_long(body)):
            suffix = "" if i == 0 else f"-p{i+1}"
            chunks.append(Chunk(
                id=f"{source}:{law[:12].replace(' ', '_')}-{sec}{suffix}", law=law, section_no=sec,
                title=title, text=part, doc_type=doc_type, source=source, official_text=True))
        cur = None

    for raw in normalize_digits(text).splitlines():
        h = HEADER_RE.match(raw.strip())
        if h:
            flush()
            law = h.group(1).strip()
            doc_type = (h.group(2) or default_type).lower()
            continue
        m = SECTION_RE.match(raw)
        if m and len(m.group(2).strip()) > 0:
            flush()
            cur = (m.group(1), m.group(2).strip(), [])
        elif cur:
            cur[2].append(raw)
    flush()
    return chunks


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # pip install pypdf. For scanned PDFs run OCR first (e.g. Tesseract 'ben').
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_seed() -> List[Chunk]:
    rows = json.loads((SEED_DIR / "seed_laws.json").read_text(encoding="utf-8"))
    return [Chunk(**r) for r in rows]


def load_raw() -> List[Chunk]:
    out: List[Chunk] = []
    for p in sorted(RAW_DIR.glob("*")):
        if p.suffix.lower() in {".txt", ".md", ".pdf"}:
            out += parse_text(read_file(p), default_law=p.stem.replace("_", " "), default_type="statute", source=p.name)
    return out


def build_chunks() -> List[Chunk]:
    """Official raw files override seed paraphrases for the same (law, section)."""
    raw = load_raw()
    official = {(c.law.lower(), c.section_no) for c in raw}
    seed = [c for c in load_seed() if (c.law.lower(), c.section_no) not in official]
    return seed + raw


def save_chunks(chunks: List[Chunk], path: Path = None):
    path = path or (INDEX_DIR / "chunks.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
