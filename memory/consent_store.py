"""Durable consent state for the live Account Aggregator flow (SQLite, stdlib only).

Why this exists: a Setu consent is approved by a HUMAN, on their own phone, minutes after the request that
created it, and the approval may arrive on a different request (or through a webhook) than the one that asked.
The graph therefore has to survive a pause and a restart. A free HF Space restarts often, so an in-memory
checkpointer (the guide's MemorySaver) would silently forget every pending consent. This store keeps just
what is needed to resume -- never bank data, and never the mobile number in clear (only a hash).

The graph is idempotent around it: aa_consent_node reads the record, aa_fetch_node runs only on ACTIVE.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import settings

TERMINAL_BAD = ("REJECTED", "REVOKED", "EXPIRED")


@dataclass(frozen=True)
class ConsentRecord:
    session_id: str
    consent_id: str
    consent_url: str
    status: str  # PENDING | ACTIVE | REJECTED | REVOKED | EXPIRED
    vua_sha256: str
    created_at: float
    updated_at: float


class ConsentStore:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path if path is not None else settings.AA_CONSENT_DB_PATH)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS consents (
                session_id TEXT PRIMARY KEY, consent_id TEXT NOT NULL UNIQUE, consent_url TEXT NOT NULL,
                status TEXT NOT NULL, vua_sha256 TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"""
        )
        self._conn.commit()

    @staticmethod
    def _row(row) -> ConsentRecord | None:
        return ConsentRecord(*row) if row else None

    def get(self, session_id: str) -> ConsentRecord | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM consents WHERE session_id=?", (session_id,)).fetchone())

    def find_by_consent(self, consent_id: str) -> ConsentRecord | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM consents WHERE consent_id=?", (consent_id,)).fetchone())

    def put(self, session_id: str, consent_id: str, consent_url: str, status: str, vua_sha256: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO consents VALUES (?,?,?,?,?,?,?)",
                (session_id, consent_id, consent_url, status, vua_sha256, now, now),
            )
            self._conn.commit()

    def update_status(self, consent_id: str, status: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            cur = self._conn.execute("UPDATE consents SET status=?, updated_at=? WHERE consent_id=?", (status, now, consent_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM consents WHERE session_id=?", (session_id,))
            self._conn.commit()

    def is_stale_pending(self, rec: ConsentRecord, now: float | None = None) -> bool:
        """A request nobody approved within the TTL is abandoned: start a fresh consent instead."""
        now = time.time() if now is None else now
        return rec.status == "PENDING" and now - rec.created_at > settings.AA_CONSENT_TTL_SECONDS


_default: ConsentStore | None = None


def default_store() -> ConsentStore:
    global _default
    if _default is None:
        _default = ConsentStore()
    return _default
