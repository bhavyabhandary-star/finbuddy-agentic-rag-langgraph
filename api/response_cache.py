"""Small TTL + LRU cache for grounded policy answers.

Why it is safe: a policy answer is a pure function of (question, language) over a FIXED
corpus, so the second person to ask "what is the DLG cap?" gets the same grounded answer
without paying for retrieval, cross-encoder re-ranking and an LLM call again.

What is never cached (see api/main.py): credit answers (they depend on the caller's own
signals), escalations, degraded / ungrounded answers (a transient failure must not stick),
and anything that used the Setu feed. Per-process memory only -- it empties on restart,
and `invalidate_all()` must be called whenever the corpus is rebuilt.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Any

from config import settings


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class ResponseCache:
    def __init__(self, ttl_seconds: float | None = None, max_entries: int | None = None):
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS
        self.max_entries = max_entries if max_entries is not None else settings.CACHE_MAX_ENTRIES
        self._data: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, query: str, language: str, now: float | None = None) -> Any | None:
        key = (normalize_query(query), language)
        now = time.monotonic() if now is None else now
        with self._lock:
            entry = self._data.get(key)
            if entry is None or now - entry[0] >= self.ttl_seconds:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, query: str, language: str, value: Any, now: float | None = None) -> None:
        key = (normalize_query(query), language)
        now = time.monotonic() if now is None else now
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def invalidate_all(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
