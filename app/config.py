"""Central configuration. Override anything via environment variables or a .env file."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"        # put official law files here (.txt / .md / .pdf)
SEED_DIR = DATA_DIR / "seed"      # starter index + category map shipped with the project
INDEX_DIR = DATA_DIR / "index"    # generated index (chunks + vectors)
EVAL_PATH = DATA_DIR / "eval" / "eval_set.json"

# "tfidf" works offline with zero downloads. "sbert" uses a multilingual embedding model (better quality).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")
SBERT_MODEL = os.getenv("SBERT_MODEL", "BAAI/bge-m3")

# "none" = deterministic template answers. "anthropic" = LLM writes a plain-language explanation.
LLM_BACKEND = os.getenv("LLM_BACKEND", "none")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CLASSIFIER_CONFIDENCE = float(os.getenv("MIN_CLASSIFIER_CONFIDENCE", "0.34"))

HELPLINES = {
    "emergency": ("999", "National Emergency Service (police, fire, ambulance)"),
    "legal_aid": ("16430", "National Legal Aid Services Organization (NLASO) helpline"),
    "women_children": ("109", "Helpline for violence against women and children"),
}

DISCLAIMER_EN = (
    "This is general legal information based on the retrieved laws, not legal advice. "
    "Laws change and every case is different - please consult a licensed lawyer or "
    "government legal aid before taking legal action."
)
DISCLAIMER_BN = (
    "এটি শুধুমাত্র সাধারণ আইনি তথ্য, আইনি পরামর্শ নয়। আইন পরিবর্তন হতে পারে এবং প্রতিটি ঘটনা আলাদা - "
    "আইনি পদক্ষেপ নেওয়ার আগে অনুগ্রহ করে একজন লাইসেন্সধারী আইনজীবী বা সরকারি আইনি সহায়তার সাথে পরামর্শ করুন।"
)
