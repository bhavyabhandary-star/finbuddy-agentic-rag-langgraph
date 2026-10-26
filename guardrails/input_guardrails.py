"""Input validation: sanitize retrieved PDF text, redact PII from the user query.

Per kickoff_prompt.md's guardrail table -- never let retrieved text carry
instructions the LLM will obey (Session 14's Bing-jailbreak case study).
"""
from __future__ import annotations

import re

# Strips characters/sequences commonly used to smuggle instructions into
# retrieved context — conservative on purpose; expand as real injection
# attempts are observed in eval-harness scenarios, not preemptively.
_CONTROL_SEQUENCE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INJECTION_MARKER_PATTERN = re.compile(
    r"(ignore (all )?(previous|prior) instructions|system prompt:|you are now)",
    re.IGNORECASE,
)
INJECTION_REPLACEMENT = "[instruction-like text removed]"

# Conservative India-context PII patterns, in the order they are applied. Aadhaar-shaped
# (12 digits) must run before the 10-digit mobile pattern or the latter would eat part of it.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("mobile", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")),
]


def sanitize_with_flags(text: str) -> tuple[str, bool]:
    """Strip control characters and NEUTRALISE injection markers in retrieved text.

    Returns (clean_text, injection_marker_found). The marker used to be detected and then
    ignored (a bare `pass`), so an instruction planted in a source document reached the
    model untouched; it is now replaced, and the flag lets the caller put it in the trace.
    """
    cleaned = _CONTROL_SEQUENCE_PATTERN.sub("", text)
    found = bool(_INJECTION_MARKER_PATTERN.search(cleaned))
    if found:
        cleaned = _INJECTION_MARKER_PATTERN.sub(INJECTION_REPLACEMENT, cleaned)
    return cleaned, found


def sanitize_retrieved_text(text: str) -> str:
    return sanitize_with_flags(text)[0]


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in _PII_PATTERNS)


_PLACEHOLDER_PATTERN = re.compile(r"\[REDACTED_[A-Z]+\]\s*")


def strip_redaction_placeholders(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _PLACEHOLDER_PATTERN.sub("", text)).strip()


def retrieval_query(text: str) -> str:
    """The text the ROUTER and the RETRIEVER should see, given an already-redacted query.

    Measured on this corpus (cross-encoder score for a well-formed DPDP-consent question,
    sufficiency threshold 0.5): the bare question 0.987; the same question after a sentence
    carrying a PAN and a phone number 0.071 with the digits left in, 0.036 with
    [REDACTED_*] tokens, 0.419 with just the tokens deleted. So the cross-encoder is badly
    thrown by a PII preamble, and redacting alone is not enough: the user would get a human
    hand-off for a question the corpus answers. This drops every sentence that contained
    PII and keeps the rest. If nothing is left (the PII was inside the question itself) it
    falls back to the sentence with only the tokens removed. The LLM still sees the full
    redacted text; only the retrievers see this.
    """
    if "[REDACTED_" not in text:
        return text
    sentences = re.split(r"(?<=[.?!])\s+|\n+", text.strip())
    kept = [s for s in sentences if "[REDACTED_" not in s]
    return " ".join(kept).strip() or strip_redaction_placeholders(text)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace phone / PAN / Aadhaar-shaped numbers with [REDACTED_<KIND>].

    Run at the API boundary BEFORE the query reaches the router, the LLM provider or the
    trace file, so a user who types their PAN into the chat box does not send it to a
    third-party model or leave it on disk. Returns (redacted_text, kinds_found).
    """
    kinds: list[str] = []
    for kind, pattern in _PII_PATTERNS:
        if pattern.search(text):
            kinds.append(kind)
            text = pattern.sub(f"[REDACTED_{kind.upper()}]", text)
    return text, kinds
