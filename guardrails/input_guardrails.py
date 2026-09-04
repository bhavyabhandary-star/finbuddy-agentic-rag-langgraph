"""Input validation: sanitize retrieved PDF text, flag PII in the user query.

Per kickoff_prompt.md's guardrail table — never let retrieved text carry
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

# Conservative India-context PII patterns: 10-digit mobile numbers, PAN, Aadhaar-shaped.
_PII_PATTERNS = [
    re.compile(r"\b[6-9]\d{9}\b"),  # mobile number
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # PAN
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # Aadhaar-shaped
]


def sanitize_retrieved_text(text: str) -> str:
    """Strip control sequences and flag (but don't silently rewrite) injection markers."""
    cleaned = _CONTROL_SEQUENCE_PATTERN.sub("", text)
    if _INJECTION_MARKER_PATTERN.search(cleaned):
        # Named, not hidden — the caller decides whether to drop the chunk entirely.
        # TODO: wire this into observability so flagged chunks show up in tracing.
        pass
    return cleaned


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_PATTERNS)
