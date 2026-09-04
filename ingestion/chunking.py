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


def semantic_chunks(text: str, source: str) -> list[RetrievedChunk]:
    """Splits on blank-line paragraph boundaries — a simple, dependency-free
    stand-in for topic/similarity-based semantic chunking.

    TODO: upgrade to embedding-similarity-based splitting once the embedding
    model from kickoff_prompt.md's Tech Stack Decisions is wired in, if fixed-size
    chunking proves too coarse on the real PDFs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [RetrievedChunk(text=p, source=source, score=0.0) for p in paragraphs]
