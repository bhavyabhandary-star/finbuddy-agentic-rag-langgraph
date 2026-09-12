"""Structured-output validation with retry-with-error-fed-back.

Per kickoff_prompt.md: forcing a Pydantic schema turns "parse the model's prose"
into "validate a structured object." On failure, retry once with the validation
error fed back to the model — don't crash on a bad parse.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

MAX_OUTPUT_VALIDATION_RETRIES = 1

# Unicode script blocks for languages generate.py can be asked to answer in.
# Only languages with a real script constraint go here -- "en" (Latin) is
# deliberately absent, since Latin script is unavoidably mixed with digits,
# punctuation and English loanwords (UPI, FinBuddy) even in a correct answer,
# making a Latin-ratio check meaningless.
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "hi": [(0x0900, 0x097F)],  # Devanagari
    "kn": [(0x0C80, 0x0CFF)],  # Kannada
}


def is_expected_script(text: str, language: str, *, min_ratio: float = 0.5) -> bool:
    """True if `language` has no script constraint, or if at least `min_ratio`
    of `text`'s alphabetic characters fall in that language's Unicode block.

    Verified necessary against real production output, not a theoretical
    guardrail: the HF-hosted fallback model (a 7B open model, weaker than
    Claude at non-English structured generation) was observed on live
    /agent/run calls answering a Kannada request in English prefixed with
    stray Han/Bopomofo characters, and a Hindi request with a leaked English
    token ("ContentLoaded") -- garbled output that reads as broken to a user,
    not just imperfect. A handful of English loanwords in an otherwise-correct
    answer are expected and fine, so this checks majority script, not purity.
    """
    ranges = _SCRIPT_RANGES.get(language)
    if not ranges:
        return True  # no script constraint for this language (e.g. English)
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True  # nothing to judge (e.g. a numbers-only answer)
    matching = sum(1 for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in ranges))
    return (matching / len(letters)) >= min_ratio


def validate_structured_output(
    raw_json: str,
    schema: type[SchemaT],
    *,
    regenerate: Callable[[str], str],
    max_retries: int = MAX_OUTPUT_VALIDATION_RETRIES,
) -> SchemaT:
    """Validates raw_json against schema; on failure, calls regenerate(error_message)
    to get a corrected completion, up to max_retries times.
    """
    attempt_json = raw_json
    last_error: ValidationError | None = None
    for _ in range(max_retries + 1):
        try:
            return schema.model_validate_json(attempt_json)
        except ValidationError as exc:
            last_error = exc
            attempt_json = regenerate(str(exc))
    assert last_error is not None
    raise last_error
