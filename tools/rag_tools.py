"""RAG tools: retrieve_pdf_chunks, check_sufficiency, fetch_source_metadata.

Per kickoff_prompt.md: hybrid dense+sparse retrieval, re-ranked, then a classical
(non-LLM) sufficiency check — the same 0.70-cosine-threshold escalation pattern
production FinBuddy's rag_service already uses.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ingestion.vector_store import VectorStore
from tools.schemas import RetrievalResult, RetrievedChunk

SUFFICIENCY_THRESHOLD = 0.70  # same gate as production FinBuddy's rag_service


def retrieve_pdf_chunks(
    query: str, vector_store: VectorStore, top_k: int = 5
) -> RetrievalResult:
    """Hybrid retrieval: dense (embedding) + sparse (BM25), then re-ranked.

    TODO: wire the real Chroma-backed dense search from vector_store.query();
    this stub currently only exercises the BM25 path so the graph is testable
    before the vector store is populated (milestone step 2).
    """
    dense_hits = vector_store.query(query, top_k=top_k)

    corpus_tokens = [c.text.split() for c in dense_hits] or [[]]
    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens != [[]] else None
    if bm25 is not None:
        bm25_scores = bm25.get_scores(query.split())
        for chunk, bm25_score in zip(dense_hits, bm25_scores):
            # simple linear blend of dense + sparse score — tune once real data exists
            chunk.score = 0.7 * chunk.score + 0.3 * float(bm25_score)

    dense_hits.sort(key=lambda c: c.score, reverse=True)
    top_score = dense_hits[0].score if dense_hits else 0.0

    return RetrievalResult(
        chunks=dense_hits,
        top_score=top_score,
        sufficient=check_sufficiency(top_score),
    )


def check_sufficiency(top_score: float, threshold: float = SUFFICIENCY_THRESHOLD) -> bool:
    """Classical threshold check — NOT an LLM judging its own retrieval (constraint 1)."""
    return top_score >= threshold


def fetch_source_metadata(chunk: RetrievedChunk) -> dict:
    """Returns citation metadata for a retrieved chunk, for the AgentResponse.sources field."""
    return {"source": chunk.source, "page": chunk.page}
