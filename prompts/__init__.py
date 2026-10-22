"""Versioned prompts -- every LLM-facing instruction lives here, not inline in node code.

Layout: prompts/<prompt_name>/<version>.txt  (v1.txt, v2.txt, ...)

Rules (a prompt change is a code change -- treat it like one):
- A published version file is IMMUTABLE. To change wording, add the next version
  file and move the pointer in ACTIVE_VERSIONS below; the old file stays in git
  history as the record of what earlier runs were told.
- tests/test_prompts.py freezes the SHA-256 of the active system prompt, so an
  in-place edit fails CI instead of silently changing model behaviour.
- The active version is written into every generate-step trace line
  (observability/tracing.py), so a run can always be tied to the exact prompt.
- Files are .txt on purpose: .dockerignore drops *.md, which would silently
  remove a .md prompt from the deployed image.

Placeholders use str.format(); literal braces are doubled ({{ }}), exactly as the
original inline templates did.
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

# The one place that decides which version of each prompt is live.
ACTIVE_VERSIONS: dict[str, str] = {
    "generate_system": "v1",
    "generate_retry_invalid_json": "v1",
    "generate_retry_wrong_script": "v1",
}


def load_prompt(name: str, version: str | None = None) -> str:
    """Returns the raw template text for `name` (active version unless one is given).

    Raises KeyError for an unknown prompt name and FileNotFoundError for a
    version that doesn't exist -- a missing prompt must fail loudly, never fall
    back to some default wording.
    """
    if name not in ACTIVE_VERSIONS:
        raise KeyError(f"Unknown prompt {name!r}; known: {sorted(ACTIVE_VERSIONS)}")
    version = version or ACTIVE_VERSIONS[name]
    path = _PROMPTS_DIR / name / f"{version}.txt"
    return path.read_text(encoding="utf-8")


def render_system_prompt(*, data: str, language_name: str) -> str:
    return load_prompt("generate_system").format(data=data, language_name=language_name)


def render_retry_suffix(name: str, **variables: str) -> str:
    """A retry instruction appended to the system prompt: blank line, then the text.

    The file's trailing newline is dropped so the result is exactly the string
    the original inline f-strings produced.
    """
    return "\n\n" + load_prompt(name).strip("\n").format(**variables)
