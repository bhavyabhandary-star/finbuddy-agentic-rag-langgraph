"""Runs the full ingestion pipeline: data/raw_pdfs/*.pdf -> Docling -> chunks
-> Chroma vector store.

Run: python -m ingestion.build_corpus
"""
from __future__ import annotations

from pathlib import Path

from ingestion.chunking import semantic_chunks
from ingestion.pdf_ingest import ingest_directory
from ingestion.vector_store import VectorStore

RAW_PDFS_DIR = Path("data/raw_pdfs")


def build_corpus() -> None:
    documents = ingest_directory(RAW_PDFS_DIR)
    store = VectorStore()
    total_chunks = 0
    for doc in documents:
        source_name = Path(doc.source_path).name
        chunks = semantic_chunks(doc.text, source=source_name)
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        print(f"{source_name}: {len(doc.text)} chars -> {len(chunks)} chunks")
    print(f"\nTotal: {len(documents)} documents, {total_chunks} chunks added to Chroma")


if __name__ == "__main__":
    build_corpus()
