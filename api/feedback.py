"""User feedback capture: the start of the learn-from-real-usage loop.

Flow:  POST /agent/feedback  ->  feedback/feedback.jsonl  ->  (a human reviews and sets
`reviewed: true`)  ->  mlops.intent_router.data.load_feedback_examples()  ->  retrain  ->
the new router must still beat the held-out gate and get a human sign-off to be promoted.

Three guards keep the loop from becoming a hole:
* feedback is only accepted for a run this process actually served (a recent-run table),
  so it can't be used to inject arbitrary text into the training file;
* every stored string is PII-redacted;
* a correction is NEVER auto-trained on -- it lands with reviewed=false.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import settings
from guardrails.input_guardrails import redact_pii

Route = Literal["policy", "credit_assessment", "off_topic"]
_RECENT_RUN_LIMIT = 2000


class FeedbackRequest(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    rating: Literal["up", "down"]
    route_was_correct: bool | None = None
    correct_route: Route | None = None
    comment: str | None = Field(default=None, max_length=settings.MAX_FEEDBACK_COMMENT_CHARS)


@dataclass(frozen=True)
class RecentRun:
    query: str  # already PII-redacted
    route: str | None
    escalated: bool


class RecentRuns:
    """Bounded run_id -> what we served. In memory: feedback on a run from before a restart is refused."""

    def __init__(self, limit: int = _RECENT_RUN_LIMIT):
        self._runs: OrderedDict[str, RecentRun] = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def remember(self, run_id: str, run: RecentRun) -> None:
        with self._lock:
            self._runs[run_id] = run
            while len(self._runs) > self._limit:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> RecentRun | None:
        with self._lock:
            return self._runs.get(run_id)


_write_lock = threading.Lock()


def append_feedback(record: dict, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else settings.FEEDBACK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def build_record(request: FeedbackRequest, run: RecentRun, route_was_correct: bool | None) -> dict:
    comment = redact_pii(request.comment)[0] if request.comment else None
    return {
        "ts": time.time(),
        "run_id": request.run_id,
        "rating": request.rating,
        "route_served": run.route,
        "escalated": run.escalated,
        "route_was_correct": route_was_correct,
        "correct_route": request.correct_route,
        "query": run.query,
        "comment": comment,
        "reviewed": False,  # a human must flip this before the example can train anything
    }
