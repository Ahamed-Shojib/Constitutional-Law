"""Module 4a - Answer generator.

Default (LLM_BACKEND=none): deterministic, citation-safe template built from retrieved provisions + category steps.
Optional (LLM_BACKEND=anthropic): an LLM writes a short plain-language explanation in the user's language,
using ONLY the retrieved provisions. Its output is then checked by guardrails.verify_explanation().
"""
import os
from typing import Dict, List, Tuple

from .config import ANTHROPIC_MODEL, HELPLINES, LLM_BACKEND, DISCLAIMER_BN, DISCLAIMER_EN
from .guardrails import verify_explanation
from .schemas import Answer, Chunk, Citation, Classification

SYSTEM_PROMPT = """You explain Bangladeshi law to ordinary people in simple words.
RULES:
- Use ONLY the provisions listed in CONTEXT. Never mention a law, section or article that is not in CONTEXT.
- Write 3-5 short sentences: what the incident most likely is legally, which provisions in CONTEXT relate to it, and why.
- Reply in the same language as the user's message (Bangla or English).
- Do not give guarantees about the outcome. Do not invent punishments or numbers.
- If CONTEXT does not fit the situation, say you are not sure and recommend a lawyer."""


def _citation(c: Chunk, score: float) -> Citation:
    snippet = c.text if len(c.text) <= 280 else c.text[:277] + "..."
    return Citation(c.label, c.law, c.section_no, c.title, snippet, c.official_text, round(score, 4))


def _llm_explain(query: str, lang: str, hits: List[Tuple[Chunk, float]]) -> str:
    import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
    ctx = "\n".join(f"- {c.label}: {c.title}. {c.text}" for c, _ in hits)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=600, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nUSER INCIDENT:\n{query}"}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _template_explain(label: str, lang: str, hits: List[Tuple[Chunk, float]], confidence: float) -> str:
    top = [c.label for c, _ in hits[:3]]
    if lang == "bn":
        return (f"আপনার বর্ণনা অনুযায়ী ঘটনাটি সম্ভবত \"{label}\" ধরনের। সবচেয়ে প্রাসঙ্গিক বিধানগুলো: "
                + "; ".join(top) + "। নিচে কী করণীয় দেওয়া হয়েছে।")
    return (f"Based on your description, this most likely falls under \"{label}\". "
            f"The most relevant provisions found are: {'; '.join(top)}. See the recommended steps below.")


def generate(query: str, lang: str, cls: Classification, hits: List[Tuple[Chunk, float]], cmap: Dict) -> Answer:
    cat = cmap["categories"].get(cls.category)
    warnings: List[str] = []
    articles = [_citation(c, s) for c, s in hits if c.doc_type == "constitution"]
    laws = [_citation(c, s) for c, s in hits if c.doc_type != "constitution"]

    if cls.confidence < 0.5 or cls.category == "unknown":
        warnings.append("The system is not fully sure about the type of incident. Please add more details (who, what, where, when).")
    if any(not c.official_text for c, _ in hits):
        warnings.append("Some provisions are short summaries, not the official wording. Verify the exact text and punishment on bdlaws.minlaw.gov.bd.")

    label = cls.label if cat else "general legal issue"
    explanation = _template_explain(label, lang, hits, cls.confidence)
    if LLM_BACKEND == "anthropic" and hits:
        try:
            raw = _llm_explain(query, lang, hits)
            checked, w = verify_explanation(raw, [c for c, _ in hits])
            warnings += w
            if checked:
                explanation = checked
        except Exception as e:  # network/key problems must never break the safe default answer
            warnings.append(f"AI explanation unavailable ({type(e).__name__}); showing the standard answer.")

    steps = list(cat["steps"]) if cat else [
        "Write down exactly what happened, when, where and who was involved, and keep any evidence.",
        "Contact the nearest police station or a lawyer. If you cannot afford one, call legal aid 16430."]
    helplines = [{"number": n, "name": d} for n, d in (HELPLINES["emergency"], HELPLINES["legal_aid"])]
    if cls.category in {"sexual_harassment_women", "rape_sexual_violence", "dowry_domestic_violence", "kidnapping"}:
        n, d = HELPLINES["women_children"]
        helplines.insert(1, {"number": n, "name": d})
    if cls.emergency:
        warnings.insert(0, "If you or someone else is in immediate danger, call 999 NOW before doing anything else.")

    return Answer(query=query, language=lang, category=cls.category, category_label=label, confidence=cls.confidence,
                  emergency=cls.emergency, explanation=explanation, constitutional_articles=articles,
                  applicable_laws=laws, steps=steps, helplines=helplines, warnings=warnings,
                  disclaimer=DISCLAIMER_BN if lang == "bn" else DISCLAIMER_EN)
