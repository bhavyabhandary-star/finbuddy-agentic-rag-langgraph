"""Ingestion: PDF (Docling) -> chunking -> Chroma vector store.

Per kickoff_prompt.md's Tech Stack Decisions: Docling for PDF text/table/image
extraction (not a web scraper — the source is real RBI/DPDP PDFs, not websites).
"""
