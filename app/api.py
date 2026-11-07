"""Module 6 - REST API.  Run: uvicorn app.api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import LegalRAG, build_index

app = FastAPI(title="Constitution of BD - Legal RAG API", version="0.1.0")
_rag = None


def rag() -> LegalRAG:
    global _rag
    if _rag is None:
        _rag = LegalRAG()
    return _rag


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Describe the incident in Bangla or English")
    top_k: int = Field(6, ge=1, le=15)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag().retriever.chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    return rag().ask(req.query, k=req.top_k).to_dict()


@app.post("/reindex")
def reindex():
    """Rebuild the index after adding official law files to data/raw/. Protect this route in production."""
    global _rag
    build_index()
    _rag = None
    return {"status": "reindexed", "chunks": len(rag().retriever.chunks)}
