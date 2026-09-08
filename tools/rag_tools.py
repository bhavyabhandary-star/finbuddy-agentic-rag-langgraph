"""RAG tools: retrieve_pdf_chunks, check_sufficiency, fetch_source_metadata.

Per kickoff_prompt.md: hybrid dense+sparse retrieval to build a candidate pool,
a cross-encoder re-ranks that pool directly on (query, chunk) pairs, and the
classical (non-LLM) sufficiency check runs on the CROSS-ENCODER's score, not
raw bi-encoder cosine similarity.

Why: verified while ingesting 5 real RBI/DPDP PDFs (milestone step 2/3) that
cosine similarity on this dense legal-text corpus topped out at ~0.45 for
genuinely correct top matches -- well under the 0.70 cutoff inherited from
production FinBuddy's shorter, conversational coaching corpus. Bi-encoder
cosine similarity is a ranking signal, not a calibrated confidence signal, and
doesn't transfer across corpora with different register/length. A quick check
against real chunks confirmed the fix: the cross-encoder scored a genuine
match at 0.996 (post-sigmoid) and irrelevant chunks at ~0.00001-0.00003 --
dramatically better separated than cosine ever was on this corpus.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.vector_store import VectorStore
from tools.schemas import RetrievalResult, RetrievedChunk

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Calibrated against real (query, chunk) pairs from this corpus (see module
# docstring) -- a true match scored 0.996, irrelevant chunks scored ~0.00003,
# so 0.5 sits with wide margin on both sides rather than splitting hairs.
SUFFICIENCY_THRESHOLD = 0.5

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def retrieve_pdf_chunks(
    query: str, vector_store: VectorStore, top_k: int = 5, rerank_pool_size: int = 15
) -> RetrievalResult:
    """Hybrid retrieval (dense + BM25) builds a candidate pool; a cross-encoder
    re-ranks that pool directly on the (query, chunk) pair. The final `score`
    on each returned chunk, and `top_score`, are the cross-encoder's sigmoid-
    mapped score, not the dense/BM25 blend used only to pick the pool.
    """
    candidates = vector_store.query(query, top_k=max(rerank_pool_size, top_k))
    if not candidates:
        return RetrievalResult(chunks=[], top_score=0.0, sufficient=False)

    corpus_tokens = [c.text.split() for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query.split())
    max_bm25 = max(bm25_scores) or 1.0
    for chunk, bm25_score in zip(candidates, bm25_scores):
        # Normalize BM25 into [0,1] before blending with cosine similarity --
        # blending an unbounded BM25 score directly against a [0,1] cosine
        # score was a real scale mismatch, fixed while wiring up re-ranking.
        chunk.score = 0.7 * chunk.score + 0.3 * (float(bm25_score) / max_bm25)
    candidates.sort(key=lambda c: c.score, reverse=True)

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    raw_scores = cross_encoder.predict(pairs)
    calibrated_scores = 1 / (1 + np.exp(-raw_scores))  # sigmoid -> [0, 1]
    for chunk, score in zip(candidates, calibrated_scores):
        chunk.score = float(score)
    candidates.sort(key=lambda c: c.score, reverse=True)

    top_chunks = candidates[:top_k]
    top_score = top_chunks[0].score

    return RetrievalResult(
        chunks=top_chunks,
        top_score=top_score,
        sufficient=check_sufficiency(top_score),
    )


def check_sufficiency(top_score: float, threshold: float = SUFFICIENCY_THRESHOLD) -> bool:
    """Classical threshold check — NOT an LLM judging its own retrieval (constraint 1)."""
    return top_score >= threshold


def fetch_source_metadata(chunk: RetrievedChunk) -> dict:
    """Returns citation metadata for a retrieved chunk, for the AgentResponse.sources field."""
    return {"source": chunk.source, "page": chunk.page}
