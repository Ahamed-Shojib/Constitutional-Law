"""Usage: python -m scripts.evaluate   -> category accuracy + Recall@k of expected provisions.
Grow data/eval/eval_set.json to 150-300 lawyer-verified scenarios for a trustworthy score."""
import json
from app.config import EVAL_PATH, TOP_K
from app.pipeline import LegalRAG

rag = LegalRAG()
rows = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
cat_ok = rec_ok = rec_total = 0
for r in rows:
    a = rag.ask(r["query"], k=TOP_K)
    got = {(c.law, c.section_no) for c in a.applicable_laws + a.constitutional_articles}
    ok_cat = a.category == r["expected_category"]
    hit = sum(tuple(e) in got for e in r["expected_sections"])
    cat_ok += ok_cat
    rec_ok += hit
    rec_total += len(r["expected_sections"])
    print(f"[{'OK ' if ok_cat else 'BAD'}] {r['query'][:60]:60s} -> {a.category:26s} provisions {hit}/{len(r['expected_sections'])}")
print(f"\nCategory accuracy: {cat_ok}/{len(rows)} = {cat_ok/len(rows):.0%}")
print(f"Provision recall@{TOP_K}: {rec_ok}/{rec_total} = {rec_ok/rec_total:.0%}")
