"""PDF analysis via Docling: text+metadata+cleanup, tables->markdown+summary,
images->detailed summary — per the Session 17 whiteboard's PDF-analysis stage.

Run: python -m ingestion.pdf_ingest data/raw_pdfs/*.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfDocument:
    source_path: str
    text: str
    tables_markdown: list[str]
    image_summaries: list[str]


def ingest_pdf(path: Path) -> PdfDocument:
    """Extracts text, tables (as markdown), and images (as summaries) from one PDF.

    TODO(milestone step 2): wire the real Docling DocumentConverter call once the
    real RBI/DPDP PDFs are collected (per build_prompt.md's resolved decision —
    do not substitute the converted markdown corpus). This stub returns an empty
    structure so downstream chunking/vector-store code is testable now.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
        text = doc.export_to_markdown()
        # TODO: split out table/image elements distinctly once real PDFs are in hand
        return PdfDocument(source_path=str(path), text=text, tables_markdown=[], image_summaries=[])
    except ImportError:
        # Docling not installed yet in this environment — fail loudly in a way
        # that's easy to diagnose rather than silently returning fake content.
        raise RuntimeError(
            "docling is not installed — run `pip install -r requirements.txt` "
            "before ingesting real PDFs."
        )


def ingest_directory(directory: Path) -> list[PdfDocument]:
    return [ingest_pdf(p) for p in sorted(directory.glob("*.pdf"))]
