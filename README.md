# ⚖️ Constitution of BD - Legal RAG for Bangladesh

Describe an incident in Bangla or English. The system classifies it, retrieves the relevant **constitutional articles**
and **statutory provisions**, and returns plain-language next steps, helplines and citations.

> Not legal advice. The Constitution defines rights; offences and punishments live in other laws (Penal Code, CrPC,
> special acts), so this project indexes both layers.

## Architecture (module by module)

```
Query (BN/EN)
  -> app/classifier.py   Module 3a  incident -> category (+ emergency flag)
  -> app/retriever.py    Module 3b  BM25 + vectors (RRF) + category hints (+ reranker hook)
        uses app/vector_store.py, app/embeddings.py, app/ingestion.py  (Module 2)
  -> app/generator.py    Module 4a  template answer, or LLM explanation (optional)
  -> app/guardrails.py   Module 4b  drops any citation not in retrieved context
  -> app/pipeline.py     Module 5   orchestration
  -> app/api.py (FastAPI) Module 6  |  ui/streamlit_app.py  Module 7
```

| Path | Purpose |
|---|---|
| `data/seed/seed_laws.json` | Starter index: 65 provisions (title + short **paraphrase**, no penalties) |
| `data/seed/category_map.json` | 17 incident categories: keywords (EN/BN), linked provisions, action steps |
| `data/raw/` | **Put official law files here** (.txt/.md/.pdf) - they override seed summaries |
| `data/eval/eval_set.json` | Test scenarios (grow to 150-300 lawyer-verified ones) |
| `scripts/build_index.py` | Build index; `scripts/evaluate.py` measures accuracy and recall |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.build_index
python -m pytest -q                  # 7 tests
python -m scripts.evaluate           # category accuracy + provision recall
streamlit run ui/streamlit_app.py    # web UI
uvicorn app.api:app --reload         # API docs: http://127.0.0.1:8000/docs
```

## Adding the real law text (most important next step)

1. Download the official texts from https://bdlaws.minlaw.gov.bd (Constitution, Penal Code, CrPC, etc.).
2. Save each as `data/raw/<name>.txt` starting with a header, then one section per numbered line:
   ```
   # LAW: Penal Code 1860 | TYPE: statute        (TYPE: constitution | statute | procedure)
   378. Theft
   Whoever, intending to take dishonestly ...
   ```
   PDFs are parsed with pypdf (scanned PDFs need OCR first, e.g. Tesseract with the `ben` language pack).
3. `python -m scripts.build_index` then `python -m scripts.evaluate`.
4. Fill in `General` entries (Cyber Security Act, Labour Act ...) with exact sections and extend `category_map.json`.

## Upgrades

| Want | Do |
|---|---|
| Better semantic search | `pip install sentence-transformers` and set `EMBEDDING_BACKEND=sbert` (BAAI/bge-m3, multilingual) |
| Plain-language LLM explanations | `pip install anthropic`, set `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY` (output is checked by guardrails) |
| Reranker | implement `HybridRetriever.rerank()` with a cross-encoder such as `BAAI/bge-reranker-v2-m3` |
| Production vector DB | replace `VectorStore` with Qdrant or pgvector (same `scores(query)` interface) |
| Smarter classifier | replace `RuleClassifier.classify()` with LLM few-shot JSON or fine-tuned BanglaBERT |
| Bangla action steps | add `steps_bn` per category (have a lawyer review the wording) |

## Safety checklist before real users

- Have a Bangladeshi lawyer review: category map, steps, helplines, and the eval set.
- Verify every helpline number and law status (acts get amended or repealed) before launch.
- Do not log personal incident details; protect `/reindex`; add rate limiting.
- Keep the disclaimer visible. Emergency cases must show 999 first.
