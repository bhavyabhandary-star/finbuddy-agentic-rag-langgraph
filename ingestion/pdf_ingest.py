"""PDF analysis via Docling: text+metadata+cleanup, tables->markdown+summary,
images->detailed summary — per the Session 17 whiteboard's PDF-analysis stage.

Verified against 5 real RBI/DPDP PDFs (milestone step 2/3) — see
data/raw_pdfs/ and build_prompt.md's resolved decision to use real government
source documents, not the converted markdown corpus.

Run: python -m ingestion.build_corpus (this module is the per-file extractor)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PdfDocument:
    source_path: str
    text: str
    tables_markdown: list[str] = field(default_factory=list)
    image_summaries: list[str] = field(default_factory=list)


def _get_converter():
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def ingest_pdf(path: Path, converter=None) -> PdfDocument:
    """Extracts text, tables (as markdown), and images (as honest, non-fabricated
    summaries) from one PDF.

    `converter` can be shared across a batch (see ingest_directory) — Docling's
    DocumentConverter loads layout/OCR model weights on construction, so
    reinstantiating it per file wastes real time and memory, verified while
    ingesting the 57-page AA Master Direction.
    """
    try:
        converter = converter or _get_converter()
    except ImportError:
        # Docling not installed — fail loudly rather than silently return fake content.
        raise RuntimeError(
            "docling is not installed — run `pip install -r requirements.txt` "
            "before ingesting real PDFs."
        )

    result = converter.convert(str(path))
    doc = result.document
    text = doc.export_to_markdown()

    tables_markdown = [table.export_to_markdown(doc) for table in doc.tables]

    # Honest placeholder, not a fabricated caption: no vision/captioning model
    # is wired in yet, and these government PDFs are text-heavy (0 tables, 1
    # picture — a letterhead logo — on the doc actually tested). Recording
    # *that* a picture exists, without inventing what it shows, matches the
    # project's "what's real vs. not yet implemented" convention.
    image_summaries = [
        f"[Image {i + 1} on page {getattr(pic.prov[0], 'page_no', '?') if pic.prov else '?'} "
        f"— no captioning model wired in yet]"
        for i, pic in enumerate(doc.pictures)
    ]

    return PdfDocument(
        source_path=str(path), text=text, tables_markdown=tables_markdown, image_summaries=image_summaries
    )


def ingest_directory(directory: Path) -> list[PdfDocument]:
    converter = _get_converter()
    return [ingest_pdf(p, converter=converter) for p in sorted(directory.glob("*.pdf"))]
