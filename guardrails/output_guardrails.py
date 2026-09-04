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
