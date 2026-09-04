"""Chroma-backed vector store — local, session-scoped ("temporary storage" per
the Session 17 whiteboard, not a permanent system of record; see kickoff_prompt.md).
"""
from __future__ import annotations

import os

from tools.schemas import RetrievedChunk

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = "finbuddy_policy_corpus"


class VectorStore:
    """Thin wrapper so tools/rag_tools.py doesn't depend on chromadb's API
    directly — swapping the backend later (kickoff_prompt.md notes pgvector as
    the noted alternative) means changing only this file.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self._persist_dir = persist_dir
        self._client = None  # lazy — see _get_client()
        self._embedding_fn = None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def _get_embedding_fn(self):
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions

            # Open-source local embeddings by default per kickoff_prompt.md's Tech
            # Stack Decisions — no external API cost for the retrieval path.
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        return self._embedding_fn

    def add_chunks(self, chunks: list[RetrievedChunk]) -> None:
        if not chunks:
            return
        collection = self._get_client().get_or_create_collection(
            COLLECTION_NAME, embedding_function=self._get_embedding_fn()
        )
        collection.add(
            ids=[f"{c.source}:{i}" for i, c in enumerate(chunks)],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page or 0} for c in chunks],
        )

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        collection = self._get_client().get_or_create_collection(
            COLLECTION_NAME, embedding_function=self._get_embedding_fn()
        )
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query_text], n_results=top_k)
        chunks = []
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma returns a distance; convert to a similarity-like score in [0, 1].
            score = 1.0 - min(distance, 1.0)
            chunks.append(RetrievedChunk(text=text, source=meta["source"], page=meta.get("page"), score=score))
        return chunks
