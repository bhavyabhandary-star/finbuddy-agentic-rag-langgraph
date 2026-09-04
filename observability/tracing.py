"""Lightweight run tracer: one JSON-lines file per run, one line per graph step.

TODO: swap for LangSmith or OpenTelemetry once an account/collector exists (see
kickoff_prompt.md's Observability section) — this stdlib version exists so tracing
is real and testable from the first commit, not deferred until infra is set up.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

TRACE_DIR = Path("./traces")


@dataclass
class RunTracer:
    run_id: str
    steps: list[dict] = field(default_factory=list)

    def log_step(self, node: str, decision: str, **details) -> None:
        """Log what a node decided and WHY — not just its output.

        Per kickoff_prompt.md's debugging-scenario section: trace tool selection
        specifically (the model's stated reason for picking a tool, not just which
        one it picked) — this is what separates a fixable bug from a mystery.
        """
        self.steps.append(
            {"timestamp": time.time(), "node": node, "decision": decision, **details}
        )

    def flush(self) -> Path:
        TRACE_DIR.mkdir(exist_ok=True)
        path = TRACE_DIR / f"{self.run_id}.jsonl"
        with open(path, "w") as f:
            for step in self.steps:
                f.write(json.dumps(step) + "\n")
        return path


@contextmanager
def trace_run(run_id: str | None = None):
    tracer = RunTracer(run_id=run_id or str(uuid.uuid4()))
    try:
        yield tracer
    finally:
        tracer.flush()
