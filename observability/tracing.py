"""Lightweight run tracer: one JSON-lines file per run, one line per graph step.

TODO: swap for LangSmith or OpenTelemetry once an account/collector exists (see
kickoff_prompt.md's Observability section) — this stdlib version exists so tracing
is real and testable from the first commit, not deferred until infra is set up.

Every string a step logs passes through redact_pii() first. The API already redacts the
user's query before the graph runs; this is the second layer, so a phone / PAN / Aadhaar
that reaches a trace some other way (a model echoing it back, a new node that logs text)
still never lands on disk.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import TRACE_DIR  # defined in config/settings.py
from guardrails.input_guardrails import redact_pii
from observability.cost_logger import RunCostTracker


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def safe_run_id(run_id: str | None) -> str:
    """run_id becomes a filename, and the API passes the client's own session_id as run_id.
    Without this, session_id="../../x" wrote outside TRACE_DIR (path traversal)."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", run_id or "")[:64]
    return cleaned or str(uuid.uuid4())


def _redact(value):
    if isinstance(value, str):
        return redact_pii(value)[0]
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    return value


@dataclass
class RunTracer:
    run_id: str
    steps: list[dict] = field(default_factory=list)
    cost: RunCostTracker = field(default_factory=RunCostTracker)
    started_at: float = field(default_factory=time.time)

    def log_step(self, node: str, decision: str, **details) -> None:
        """Log what a node decided and WHY — not just its output.

        Per kickoff_prompt.md's debugging-scenario section: trace tool selection
        specifically (the model's stated reason for picking a tool, not just which
        one it picked) — this is what separates a fixable bug from a mystery.
        """
        self.cost.record_step()
        self.steps.append(
            {"timestamp": time.time(), "node": node, "decision": _redact(decision), **_redact(details)}
        )

    def summary(self) -> dict:
        return {
            "node": "run_summary",
            "steps": self.cost.steps,
            "wall_seconds": round(time.time() - self.started_at, 3),
            **self.cost.summary(),
        }

    def flush(self) -> Path:
        TRACE_DIR.mkdir(exist_ok=True)
        path = TRACE_DIR / f"{safe_run_id(self.run_id)}.jsonl"
        with open(path, "w") as f:
            for step in self.steps:
                f.write(json.dumps(step) + "\n")
            f.write(json.dumps(self.summary()) + "\n")
        return path


@contextmanager
def trace_run(run_id: str | None = None):
    tracer = RunTracer(run_id=safe_run_id(run_id))
    try:
        yield tracer
    finally:
        tracer.flush()
