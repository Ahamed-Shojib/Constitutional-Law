import json

from app.config import EVAL_PATH
from app.guardrails import verify_explanation
from app.ingestion import parse_text
from app.pipeline import LegalRAG
from app.schemas import Chunk
from app.classifier import detect_language


def rag_fixture():
    return LegalRAG()


def test_parse_text_sections_and_bangla_digits():
    txt = "# LAW: Test Act 2000 | TYPE: statute\n১০. Title ten\nBody of ten.\n11. Title eleven\nBody of eleven."
    chunks = parse_text(txt, "x", "statute", "t.txt")
    assert [c.section_no for c in chunks] == ["10", "11"]
    assert chunks[0].law == "Test Act 2000" and "Body of ten" in chunks[0].text


def test_language_detection():
    assert detect_language("আমার মোবাইল চুরি হয়েছে") == "bn"
    assert detect_language("my phone was stolen") == "en"


def test_guardrail_removes_unretrieved_citation():
    hits = [Chunk("a", "Penal Code 1860", "379", "Theft", "x")]
    text = "Section 379 applies here. Section 999 also applies."
    out, warns = verify_explanation(text, hits)
    assert "379" in out and "999" not in out and len(warns) == 1


def test_theft_example_returns_theft_provisions():
    a = rag_fixture().ask("Someone snatched my phone on the road")
    assert a.category == "theft_robbery"
    assert any(c.section_no in {"378", "379", "390", "392"} for c in a.applicable_laws)
    assert a.steps and a.helplines and a.disclaimer


def test_arrest_example_returns_article_33():
    a = rag_fixture().ask("Police arrested my brother and he was not produced before a magistrate")
    assert any(c.section_no == "33" for c in a.constitutional_articles)


def test_unknown_query_is_safe():
    a = rag_fixture().ask("what is the weather")
    assert a.category == "unknown" and a.warnings and a.disclaimer


def test_eval_set_thresholds():
    rag = rag_fixture()
    rows = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    ok = sum(rag.ask(r["query"]).category == r["expected_category"] for r in rows)
    assert ok / len(rows) >= 0.8
