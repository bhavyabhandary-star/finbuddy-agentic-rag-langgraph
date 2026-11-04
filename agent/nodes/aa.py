"""Live Account Aggregator path: consent -> (pause) -> fetch, ahead of the existing credit assessment.

    route(credit, aa_mode) -> aa_consent -+- ACTIVE  -> aa_fetch -+- ok    -> credit_assessment -> generate
                                          |                       +- error -> escalate
                                          +- PENDING -> aa_pending (END: "approve the request, then ask again")
                                          +- REJECTED / REVOKED / EXPIRED / not configured -> escalate

Design rules, each one a deliberate difference from a naive "put the bank JSON in an LLM prompt" design:

* RAW BANK DATA NEVER LEAVES aa_fetch_node. The payload is normalised to the same 8 numeric signals the scoring API
  takes and the raw session is dropped. It is not put in graph state (so no checkpoint), not traced, not sent to an LLM.
  Only the 8 numbers continue, through the existing, already-governed credit path.
* The mobile / AA handle is used once, to create the consent, and stored only as a hash.
* A webhook is a hint, not an authority (api/main.py re-reads the status from Setu before trusting it).
* Every failure is a fixed, honest hand-off. Nothing here can produce a score from data we did not get.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from agent.state import AgentState
from config import settings
from memory.consent_store import TERMINAL_BAD, ConsentStore, default_store
from mlops.real_data.constants import SIGNALS
from observability.tracing import RunTracer

logger = logging.getLogger("agent.aa")

_pending_message = {
    "en": (
        "To check your credit profile I need your permission to read your bank data through an Account Aggregator. "
        "Please approve the data-sharing request using the link provided, then ask again. I fetch nothing until you approve."
    ),
    "hi": (
        "आपकी क्रेडिट प्रोफ़ाइल जाँचने के लिए मुझे Account Aggregator के ज़रिए आपका बैंक डेटा देखने की अनुमति चाहिए। "
        "कृपया दिए गए लिंक से अनुरोध स्वीकार करें और फिर दोबारा पूछें। आपकी स्वीकृति से पहले मैं कुछ नहीं लेती।"
    ),
}
_error_message = {
    "en": (
        "I couldn't get your bank data through the Account Aggregator (the request was declined or expired, wasn't "
        "available, or the service isn't set up here), so I can't give you an assessment. A human coach can help directly."
    ),
    "hi": (
        "मैं Account Aggregator के ज़रिए आपका बैंक डेटा नहीं ला पाई (अनुरोध अस्वीकार या समाप्त हुआ, डेटा उपलब्ध नहीं था, या यहाँ सेवा "
        "चालू नहीं है), इसलिए आकलन नहीं दे सकती। कोई इंसान कोच सीधे मदद कर सकता है।"
    ),
}


def pending_message(language: str) -> str:
    return _pending_message.get(language, _pending_message["en"])  # kn: English, as the project's Kannada policy


def error_message(language: str) -> str:
    return _error_message.get(language, _error_message["en"])


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def default_client():
    """The real Setu client, or None when SETU_* credentials are not configured (the deployed demo has none)."""
    from ingestion.setu_feed import SetuAAConfigError, SetuAAFeedClient

    try:
        return SetuAAFeedClient()
    except SetuAAConfigError:
        return None


def _normalise_status(raw: str | None) -> str:
    raw = (raw or "").upper()
    if raw == "ACTIVE":
        return "ACTIVE"
    if raw in TERMINAL_BAD:
        return raw
    return "PENDING"


def aa_consent_node(state: AgentState, client=None, store: ConsentStore | None = None, tracer: RunTracer | None = None) -> AgentState:
    store = store or default_store()
    session_id = state["session_id"]
    record = store.get(session_id)

    def done(status: str, record_id: str | None = None, url: str | None = None, error: str | None = None) -> AgentState:
        if tracer:
            tracer.log_step("aa_consent", decision=f"consent {status}", error=error)
        return {**state, "consent_status": status, "consent_id": record_id, "consent_url": url, "aa_error": error}

    try:
        if record is not None and not store.is_stale_pending(record):
            status = record.status
            if status == "PENDING":  # a missed webhook must not strand the user: re-read from Setu
                client = client or default_client()
                if client is not None:
                    status = _normalise_status(client.get_consent_status(record.consent_id).get("status"))
                    if status != record.status:
                        store.update_status(record.consent_id, status)
            return done(status, record.consent_id, record.consent_url, None if status in ("PENDING", "ACTIVE") else f"consent_{status.lower()}")

        handle = state.get("aa_handle")
        if not handle:
            return done("FAILED", error="aa_handle_missing")
        client = client or default_client()
        if client is None:
            return done("FAILED", error="aa_not_configured")
        now = datetime.now(timezone.utc)
        consent = client.create_consent(
            vua=handle,
            duration_months=settings.AA_CONSENT_DURATION_MONTHS,
            data_range_from=_iso(now - timedelta(days=30 * settings.AA_CONSENT_DURATION_MONTHS)),
            data_range_to=_iso(now),
        )
        store.put(session_id, consent["id"], consent["url"], "PENDING", hashlib.sha256(handle.encode("utf-8")).hexdigest())
        return done("PENDING", consent["id"], consent["url"])
    except Exception as exc:  # noqa: BLE001 -- any gateway failure is a safe hand-off, never a 500
        logger.warning("aa_consent_failed", extra={"error_type": type(exc).__name__})
        return done("FAILED", error=f"aa_gateway_error:{type(exc).__name__}")


def aa_fetch_node(state: AgentState, client=None, tracer: RunTracer | None = None) -> AgentState:
    def fail(reason: str) -> AgentState:
        if tracer:
            tracer.log_step("aa_fetch", decision="fetch failed", error=reason)
        return {**state, "aa_error": reason}

    client = client or default_client()
    if client is None:
        return fail("aa_not_configured")
    try:
        now = datetime.now(timezone.utc)
        data_from = _iso(now - timedelta(days=30 * settings.AA_CONSENT_DURATION_MONTHS))
        data_to = _iso(now)
        # The data window must sit inside the range Setu fixed when the consent was created; "now" is already past its
        # end, so ask for the granted range itself. (Fall back to the computed window only if Setu does not state one.)
        granted = (client.get_consent_status(state["consent_id"]).get("detail") or {}).get("dataRange") or {}
        if granted.get("from") and granted.get("to"):
            data_from, data_to = granted["from"], granted["to"]
        session = client.create_data_session(state["consent_id"], data_from, data_to)
        completed = client.wait_for_data_session(session["id"], poll_seconds=2, timeout_seconds=settings.AA_DATA_FETCH_TIMEOUT_SECONDS)
        from ingestion.setu_feed import normalize_session_response

        profiles = normalize_session_response(completed)
        del completed  # the raw bank payload ends here
    except Exception as exc:  # noqa: BLE001
        return fail(f"aa_fetch_error:{type(exc).__name__}")
    if not profiles:
        return fail("aa_no_deposit_accounts")
    signals = {name: profiles[0][name] for name in SIGNALS}
    if tracer:
        tracer.log_step("aa_fetch", decision="normalised to 8 signals; raw payload discarded", accounts_used=1)
    return {**state, "credit_signals": signals, "aa_error": None}


def aa_pending_node(state: AgentState, tracer: RunTracer | None = None) -> AgentState:
    language = state.get("response_language") or "en"
    if tracer:
        tracer.log_step("aa_pending", decision="waiting for the user to approve consent", routed_to_human=False)
    return {
        **state,
        "answer": pending_message(language),
        "sources": [],
        "confidence": 0.0,
        "escalate_to_human": False,
        "disclaimer": None,
    }
