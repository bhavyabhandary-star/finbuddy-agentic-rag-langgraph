"""Chunking: fixed-size (token/word-based, overlapping) and semantic (paragraph-
based) strategies — per Session 11's Document Processing material.
"""
from __future__ import annotations

from tools.schemas import RetrievedChunk


def fixed_size_chunks(
    text: str, source: str, chunk_words: int = 200, overlap_words: int = 40
) -> list[RetrievedChunk]:
    words = text.split()
    chunks: list[RetrievedChunk] = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(RetrievedChunk(text=" ".join(window), source=source, score=0.0))
        if start + chunk_words >= len(words):
            break
    return chunks


MIN_CHUNK_WORDS = 5  # drops structural noise (headers, "## Illustration.",
# gazette boilerplate, bare cross-reference citations like "24 of 1997.") --
# verified against the real DPDP_Act_2023.pdf markdown: every one of its 55
# sub-5-word paragraphs was exactly this kind of noise, none real content.


def semantic_chunks(text: str, source: str) -> list[RetrievedChunk]:
    """Splits on blank-line paragraph boundaries -- Docling's markdown export
    already isolates each numbered clause/sub-section into its own paragraph
    for these government PDFs (verified directly against the real extracted
    text), so this aligns chunk boundaries with actual section boundaries
    instead of cutting mid-clause.

    Replaced fixed_size_chunks as build_corpus.py's default after a real
    RAGAS eval run surfaced the failure mode this was meant to prevent: a
    200-word sliding window spliced the tail of one DPDP_Act_2023.pdf clause
    together with the start of an unrelated one (Section 6's consent
    definition), diluting that chunk's embedding relevance enough that it
    never ranked in the top-15 candidates for a query that should have
    surfaced it. See docs/ragas_eval_results.json for the original finding.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if len(p.split()) >= MIN_CHUNK_WORDS]
    return [RetrievedChunk(text=p, source=source, score=0.0) for p in paragraphs]
