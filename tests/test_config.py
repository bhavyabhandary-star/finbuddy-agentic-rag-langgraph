"""config/ guard tests: values are unchanged by the move, and tunables stay in config.

The "no stray tunables" scan is the point of the layout rule: if someone adds a
new module-level constant to generate.py / rag_tools.py / etc. instead of
config/settings.py, this fails and says where it belongs.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modules that used to define these constants inline. They may import them
# (re-exporting the same name), but must not assign them again.
MODULES_THAT_MUST_NOT_DEFINE_TUNABLES = [
    "agent/state.py",
    "agent/nodes/generate.py",
    "agent/graph.py",
    "tools/rag_tools.py",
    "tools/credit_tools.py",
    "guardrails/tool_policy.py",
    "guardrails/output_guardrails.py",
    "observability/tracing.py",
    "eval/evaluate_agent.py",
    "api/main.py",
]


def _config_names() -> set[str]:
    return {n for n in vars(settings) if n.isupper() and not n.startswith("_")}


def _top_level_assigned_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@pytest.mark.parametrize("relpath", MODULES_THAT_MUST_NOT_DEFINE_TUNABLES)
def test_tunables_are_not_redefined_outside_config(relpath):
    clash = _top_level_assigned_names(PROJECT_ROOT / relpath) & _config_names()
    assert not clash, f"{relpath} assigns {sorted(clash)} itself -- define it in config/settings.py and import it"


def test_values_are_unchanged_by_the_move():
    """The numbers/strings that used to live inline, frozen so a 'refactor' can't drift them."""
    assert settings.MAX_OUTPUT_TOKENS == 512
    assert settings.MAX_LOOP_ITERATIONS == 3
    assert settings.SUFFICIENCY_THRESHOLD == 0.5
    assert settings.CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.MAX_RETRIES_PER_TOOL_CALL == 1
    assert settings.MAX_OUTPUT_VALIDATION_RETRIES == 1
    assert settings.GATE_MIN_ROUTING_ACCURACY == 0.75
    assert settings.DEFAULT_CLAUDE_MODEL == "claude-sonnet-5"
    assert settings.DEFAULT_HF_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert settings.DEFAULT_HF_PROVIDER == "featherless-ai"
    assert settings.DEFAULT_OLLAMA_MODEL == "llama3.1"
    assert settings.LANGUAGE_NAMES == {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    assert settings.SCORING_API_TIMEOUT_SECONDS == 60.0


def test_old_import_paths_still_work_and_agree():
    """Existing code/tests import these names from their old modules."""
    from agent.nodes.generate import DEFAULT_HF_MODEL, LANGUAGE_NAMES, MAX_OUTPUT_TOKENS
    from agent.state import MAX_LOOP_ITERATIONS
    from eval.evaluate_agent import GATE_MIN_ROUTING_ACCURACY
    from guardrails.output_guardrails import MAX_OUTPUT_VALIDATION_RETRIES
    from guardrails.tool_policy import MAX_RETRIES_PER_TOOL_CALL
    from tools.credit_tools import RISK_TREND_ARTIFACT_PATH, SCORING_API_BASE_URL
    from tools.rag_tools import CROSS_ENCODER_MODEL, SUFFICIENCY_THRESHOLD

    assert DEFAULT_HF_MODEL == settings.DEFAULT_HF_MODEL
    assert LANGUAGE_NAMES is settings.LANGUAGE_NAMES
    assert MAX_OUTPUT_TOKENS == settings.MAX_OUTPUT_TOKENS
    assert MAX_LOOP_ITERATIONS == settings.MAX_LOOP_ITERATIONS
    assert GATE_MIN_ROUTING_ACCURACY == settings.GATE_MIN_ROUTING_ACCURACY
    assert MAX_OUTPUT_VALIDATION_RETRIES == settings.MAX_OUTPUT_VALIDATION_RETRIES
    assert MAX_RETRIES_PER_TOOL_CALL == settings.MAX_RETRIES_PER_TOOL_CALL
    assert RISK_TREND_ARTIFACT_PATH == settings.RISK_TREND_ARTIFACT_PATH
    assert SCORING_API_BASE_URL == settings.SCORING_API_BASE_URL
    assert CROSS_ENCODER_MODEL == settings.CROSS_ENCODER_MODEL
    assert SUFFICIENCY_THRESHOLD == settings.SUFFICIENCY_THRESHOLD


def test_call_time_getters_read_the_environment(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.delenv("HF_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert settings.llm_provider_name() == "claude"
    assert settings.hf_model() == settings.DEFAULT_HF_MODEL
    assert settings.hf_provider() == settings.DEFAULT_HF_PROVIDER
    assert settings.ollama_model() == settings.DEFAULT_OLLAMA_MODEL
    assert settings.hf_token() is None  # secrets never get a default

    monkeypatch.setenv("LLM_PROVIDER", "HuggingFace")
    monkeypatch.setenv("HF_MODEL", "some/other-model")
    monkeypatch.setenv("HF_TOKEN", "x")
    assert settings.llm_provider_name() == "huggingface"  # lower-cased, as api/main.py always did
    assert settings.hf_model() == "some/other-model"
    assert settings.hf_token() == "x"


def test_no_secret_has_a_default_in_config():
    """Keys live in the environment, never in code."""
    source = (PROJECT_ROOT / "config" / "settings.py").read_text(encoding="utf-8")
    for secret_env in ("ANTHROPIC_API_KEY", "HF_TOKEN", "SETU_CLIENT_SECRET", "LANGCHAIN_API_KEY"):
        assert f'environ.get("{secret_env}",' not in source, f"{secret_env} must not have an in-code default"
