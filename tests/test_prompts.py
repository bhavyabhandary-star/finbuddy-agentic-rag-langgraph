"""prompts/ guard tests: the move out of generate.py changed nothing the model sees,
and published prompt versions can't be edited in place.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.nodes.generate import SYSTEM_PROMPT_TEMPLATE, generate_node
from observability.tracing import RunTracer
from prompts import ACTIVE_VERSIONS, load_prompt, render_retry_suffix, render_system_prompt

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# SHA-256 of the ORIGINAL inline SYSTEM_PROMPT_TEMPLATE (generate.py at the commit
# before prompts/ existed). A published version is immutable: changing wording
# means adding v2.txt and moving ACTIVE_VERSIONS, not editing v1.txt.
V1_SYSTEM_SHA256 = "7df3190ce0b150fede80a23a9a1fd48d69615dfcf88cb14d46c065a6679e590b"


def test_v1_system_prompt_is_frozen():
    text = (PROMPTS_DIR / "generate_system" / "v1.txt").read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == V1_SYSTEM_SHA256


def test_generate_module_still_exposes_the_template():
    assert SYSTEM_PROMPT_TEMPLATE == load_prompt("generate_system")


def test_rendered_system_prompt_matches_the_original_inline_behaviour():
    rendered = render_system_prompt(data='{"x": 1}', language_name="Hindi")
    assert rendered == SYSTEM_PROMPT_TEMPLATE.format(data='{"x": 1}', language_name="Hindi")
    assert "Write the \"answer\" field in Hindi" in rendered
    assert '{"answer": "...", "confidence": 0.0, "escalate_to_human": false}' in rendered  # braces un-doubled


def test_retry_suffixes_match_the_original_inline_f_strings():
    error_message = "1 validation error {with braces}"
    language_name = "Hindi"
    # exactly what generate_node used to build inline:
    original_invalid = f"\n\nYour previous output was invalid: {error_message}\nReturn ONLY valid JSON."
    original_wrong_script = (
        f"\n\nYour previous answer was not written in {language_name}. "
        f"Rewrite the answer fully in {language_name}."
    )
    assert render_retry_suffix("generate_retry_invalid_json", error_message=error_message) == original_invalid
    assert render_retry_suffix("generate_retry_wrong_script", language_name=language_name) == original_wrong_script


def test_every_active_version_file_exists():
    for name, version in ACTIVE_VERSIONS.items():
        assert (PROMPTS_DIR / name / f"{version}.txt").is_file(), f"{name}/{version}.txt is missing"


def test_prompt_files_are_txt_so_dockerignore_cannot_drop_them():
    """.dockerignore excludes *.md -- a .md prompt would vanish from the deployed image."""
    files = [p for p in PROMPTS_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for p in files:
        assert p.suffix in {".txt", ".py"}, f"{p} would be dropped or mis-packaged; prompts must be .txt"


def test_unknown_prompt_and_missing_version_fail_loudly():
    with pytest.raises(KeyError):
        load_prompt("does_not_exist")
    with pytest.raises(FileNotFoundError):
        load_prompt("generate_system", version="v999")


class _FakeProvider:
    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        return '{"answer": "ok", "sources": [], "confidence": 0.9, "escalate_to_human": false}'


def test_generate_step_trace_records_the_prompt_version():
    tracer = RunTracer(run_id="t")
    state = {"query": "q", "route": "policy", "retrieved_chunks": [], "sufficient": True, "response_language": "en"}
    generate_node(state, provider=_FakeProvider(), tracer=tracer)
    generate_steps = [s for s in tracer.steps if s["node"] == "generate"]
    assert generate_steps and generate_steps[-1]["prompt_version"] == ACTIVE_VERSIONS["generate_system"]
