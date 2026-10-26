"""API access control: optional API key + per-client rate limit.

Both are FastAPI dependencies applied to every /agent/* route and /metrics (not /health).

* API key: FINBUDDY_API_KEY unset = open demo API, exactly as deployed before this module
  existed (the GitHub Pages console has no way to keep a secret). Set it and every call
  must send `X-API-Key`. Compared in constant time.
* Rate limit: an in-memory sliding window per client. Per-process, so with several worker
  processes the effective limit is multiplied -- a real deployment would put this in the
  gateway / Redis. It still stops the failure that matters here: one client in a loop
  burning the LLM quota.

NOT solved here, and said plainly: the API does not know WHO the caller is, so it cannot
bind `credit_signals` to a verified person. The signals are caller-supplied (or the cached
Setu sandbox profile). That needs real authentication in front of it before this ever
scores a real borrower.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from config import settings


def require_api_key(request: Request) -> None:
    expected = settings.api_key()
    if expected is None:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def client_id(request: Request) -> str:
    if settings.trust_proxy_headers():
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self, max_requests: int | None = None, window_seconds: float | None = None):
        self.max_requests = max_requests if max_requests is not None else settings.RATE_LIMIT_MAX_REQUESTS
        self.window_seconds = window_seconds if window_seconds is not None else settings.RATE_LIMIT_WINDOW_SECONDS
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float | None:
        """Records a hit. Returns None if allowed, else seconds until the client may retry."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return max(0.0, self.window_seconds - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under an address-spraying client
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window_seconds}
            return None


_limiter = RateLimiter()


def enforce_rate_limit(request: Request) -> None:
    retry_after = _limiter.check(client_id(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
