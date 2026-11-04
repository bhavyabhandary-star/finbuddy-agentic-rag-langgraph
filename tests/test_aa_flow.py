"""The live Account Aggregator path through the real LangGraph + real FastAPI, with a fake Setu gateway.

What must hold: no score is ever produced from data we did not get; the person's approval is a hard gate;
the raw bank payload never reaches state, traces or the LLM; a forged webhook cannot approve a consent;
every failure is a fixed hand-off; and an unconfigured deployment (like the public demo) fails safe.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import agent.nodes.aa as aa_module
import agent.nodes.credit_assessment as credit_module
import agent.nodes.route as route_module
import api.main as main
import observability.tracing as tracing
from api import security
from api.response_cache import ResponseCache
from config import settings
from memory.consent_store import ConsentStore
from observability.metrics import metrics
from tools.schemas import CreditAssessmentResult, ScoreFactor

HANDLE = "9999999999@onemoney"
SECRET_NARRATION = "SALARY ACME PRIVATE LIMITED CONFIDENTIAL"


def make_session() -> dict:
    """A completed Setu data session: 12 months of one deposit account, shaped like the real response."""
    txns = []
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    balance = 30000.0
    for m in range(12):
        t = start + timedelta(days=30.4 * m)
        balance += 20000
        txns.append({"transactionTimestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "amount": "20000", "type": "CREDIT",
                     "narration": SECRET_NARRATION, "currentBalance": str(balance)})
        for d in range(4):
            balance -= 3500
            txns.append({"transactionTimestamp": (t + timedelta(days=2 + d)).strftime("%Y-%m-%dT%H:%M:%SZ"), "amount": "3500",
                         "type": "DEBIT", "narration": f"UPI/merchant{d}/groceries", "currentBalance": str(balance)})
    return {"id": "s-1", "status": "COMPLETED", "fips": [{"accounts": [{"FIstatus": "READY", "linkRefNumber": "acct-ref-1",
            "data": {"account": {"type": "deposit", "transactions": {"transaction": txns}}}}]}]}


class FakeSetu:
    def __init__(self):
        self.status = "PENDING"
        self.created = 0
        self.status_lookups = 0
        self.fetch_error: Exception | None = None
        self.granted_range: dict | None = None
        self.session_ranges: list[tuple[str, str]] = []

    def create_consent(self, vua, duration_months=12, data_range_from=None, data_range_to=None):
        self.created += 1
        return {"id": f"consent-{self.created}", "url": f"https://fake.setu/approve/{self.created}", "status": "PENDING"}

    def get_consent_status(self, consent_id):
        self.status_lookups += 1
        body = {"id": consent_id, "status": self.status}
        if self.granted_range:
            body["detail"] = {"dataRange": self.granted_range}  # where the real Setu states the window it fixed at creation
        return body

    def create_data_session(self, consent_id, data_range_from, data_range_to):
        self.session_ranges.append((data_range_from, data_range_to))
        return {"id": "s-1"}

    def wait_for_data_session(self, session_id, poll_seconds=5, timeout_seconds=300):
        if self.fetch_error:
            raise self.fetch_error
        return make_session()


class StubProvider:
    def __init__(self):
        self.prompts: list[str] = []
        self.last_usage = None

    def complete(self, system_prompt, user_query, max_tokens):
        self.prompts.append(system_prompt + "\n" + user_query)
        return json.dumps({"answer": "Your credit score is 700, which is good.", "confidence": 0.9, "escalate_to_human": False})


FACTORS = [ScoreFactor(factor="income_regularity_score", shap_contribution=0.1, direction="positive",
                       plain_english="Regular income.", action="Keep it up.")]


def fake_assess(signals):
    fake_assess.seen = dict(signals)
    return CreditAssessmentResult(credit_score=700, calibrated_probability_of_repayment=0.7, approved=True,
                                  fairness_mitigation_applied=True, income_band="middle", is_anomalous=False,
                                  anomaly_note=None, top_3_factors=FACTORS, latency_ms=10.0)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_WARMUP", "1")
    monkeypatch.delenv("FINBUDDY_API_KEY", raising=False)
    monkeypatch.delenv("SETU_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(route_module, "classify_intent", lambda q: ("credit_assessment", 0.9))
    monkeypatch.setattr(credit_module, "assess_credit_profile", fake_assess)
    monkeypatch.setattr(tracing, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(security, "_limiter", security.RateLimiter(1000, 60))
    monkeypatch.setattr(main, "_cache", ResponseCache())
    metrics.reset()
    fake, store, provider = FakeSetu(), ConsentStore(":memory:"), StubProvider()
    monkeypatch.setitem(main._AA, "client", fake)
    monkeypatch.setitem(main._AA, "store", store)
    monkeypatch.setattr(main, "_default_provider", lambda: provider)
    return {"client": TestClient(main.app), "setu": fake, "store": store, "provider": provider, "tmp": tmp_path}


def ask(env, session="sess-1", **extra):
    body = {"query": "can you check my credit score", "session_id": session, "aa_handle": HANDLE, **extra}
    return env["client"].post("/agent/run", json=body).json()


# ---------------------------------------------------------------- consent is a hard gate


def test_first_request_creates_a_consent_and_pauses_without_a_score_or_an_llm_call(env):
    body = ask(env)
    assert body["consent_status"] == "PENDING" and body["consent_url"] == "https://fake.setu/approve/1"
    assert not body["escalate_to_human"] and body["credit_assessment"] is None and body["failure_reason"] is None
    assert "approve" in body["answer"].lower()
    assert env["provider"].prompts == [] and env["setu"].created == 1


def test_asking_again_before_approval_does_not_create_a_second_consent_or_fetch_data(env):
    ask(env)
    body = ask(env)
    assert body["consent_status"] == "PENDING" and env["setu"].created == 1 and body["credit_assessment"] is None


def test_after_approval_the_data_is_fetched_normalised_and_scored(env):
    ask(env)
    env["setu"].status = "ACTIVE"  # the person approved on their phone; no webhook arrived
    body = ask(env)
    assert body["consent_status"] == "ACTIVE" and body["credit_assessment"]["credit_score"] == 700
    assert body["answer"].startswith("Your credit score") and body["disclaimer"]
    assert set(fake_assess.seen) == {"avg_monthly_income", "income_regularity_score", "tx_count_30d", "merchant_diversity",
                                     "balance_dip_frequency", "b2b_ratio", "avg_transaction_size", "tenure_months"}
    assert 15000 < fake_assess.seen["avg_monthly_income"] < 25000  # ~20,000 of salary a month, give or take month bucketing


def test_the_data_window_is_the_range_the_consent_was_granted_not_today(env):
    # Setu rejects a window that runs past the consent's own range ("FI dataRange is not within the consent's FIDataRange").
    ask(env)
    env["setu"].status = "ACTIVE"
    env["setu"].granted_range = {"from": "2025-11-09T21:30:09.000Z", "to": "2026-11-04T21:30:09.000Z"}
    body = ask(env)
    assert env["setu"].session_ranges == [("2025-11-09T21:30:09.000Z", "2026-11-04T21:30:09.000Z")]
    assert body["credit_assessment"]["credit_score"] == 700


def test_a_declined_consent_ends_in_a_handoff_with_no_score(env):
    ask(env)
    env["setu"].status = "REJECTED"
    body = ask(env)
    assert body["consent_status"] == "REJECTED" and body["escalate_to_human"] and body["credit_assessment"] is None
    assert env["provider"].prompts == []


def test_an_abandoned_pending_consent_is_replaced_not_reused_forever(env, monkeypatch):
    ask(env)
    monkeypatch.setattr(settings, "AA_CONSENT_TTL_SECONDS", -1)  # everything pending is now "older than the TTL"
    body = ask(env)
    assert env["setu"].created == 2 and body["consent_id"] == "consent-2"


# ---------------------------------------------------------------- the bank payload stays out of everything


def test_raw_bank_data_never_reaches_the_llm_the_traces_or_the_response(env):
    ask(env)
    env["setu"].status = "ACTIVE"
    body = ask(env)
    assert SECRET_NARRATION.lower() not in json.dumps(body).lower()
    assert all(SECRET_NARRATION.lower() not in p.lower() for p in env["provider"].prompts)
    traces = "".join(p.read_text() for p in (env["tmp"] / "traces").glob("*.jsonl"))
    assert traces and SECRET_NARRATION.lower() not in traces.lower() and "9999999999" not in traces


def test_the_mobile_is_stored_only_as_a_hash(env):
    ask(env)
    rec = env["store"].get("sess-1")
    assert rec.vua_sha256 and HANDLE not in repr(rec) and "9999999999" not in repr(rec)


# ---------------------------------------------------------------- failures are safe hand-offs


def test_a_gateway_failure_while_fetching_is_a_handoff(env):
    ask(env)
    env["setu"].status = "ACTIVE"
    env["setu"].fetch_error = TimeoutError("slow")
    body = ask(env)
    assert body["escalate_to_human"] and body["credit_assessment"] is None
    assert "bank data" in body["answer"]


def test_an_unconfigured_deployment_like_the_public_demo_fails_safe(env, monkeypatch):
    monkeypatch.setitem(main._AA, "client", None)
    monkeypatch.setattr(aa_module, "default_client", lambda: None)
    body = ask(env)
    assert body["escalate_to_human"] and body["consent_status"] == "FAILED" and body["credit_assessment"] is None


def test_the_cache_never_serves_or_stores_an_aa_request(env):
    ask(env)
    assert len(main._cache) == 0


@pytest.mark.parametrize("bad", ["not-a-handle", "12345@onemoney", "99999999999", "9999999999@", "9999999999@x y"])
def test_malformed_aa_handles_are_rejected_at_the_boundary(env, bad):
    r = env["client"].post("/agent/run", json={"query": "check my score", "aa_handle": bad})
    assert r.status_code == 422


@pytest.mark.parametrize("ok", ["9999999999", "9999999999@finvu"])
def test_a_bare_mobile_or_a_mobile_with_a_handle_are_both_accepted(env, ok):
    body = ask(env, aa_handle=ok)
    assert body["consent_status"] == "PENDING"


# ---------------------------------------------------------------- webhook


def webhook(env, body, secret=None):
    headers = {"X-Webhook-Secret": secret} if secret is not None else {}
    return env["client"].post("/webhooks/setu/consent", json=body, headers=headers)


def test_the_webhook_does_not_exist_until_a_secret_is_configured(env):
    assert webhook(env, {"consentId": "consent-1"}).status_code == 404


def test_the_webhook_rejects_a_wrong_secret(env, monkeypatch):
    monkeypatch.setenv("SETU_WEBHOOK_SECRET", "s3cret")
    assert webhook(env, {"consentId": "consent-1"}, secret="nope").status_code == 401
    assert webhook(env, {"consentId": "consent-1"}).status_code == 401


def test_the_webhook_ignores_a_consent_we_never_created(env, monkeypatch):
    monkeypatch.setenv("SETU_WEBHOOK_SECRET", "s3cret")
    assert webhook(env, {"consentId": "someone-elses"}, secret="s3cret").json() == {"status": "ignored"}


def test_a_forged_webhook_cannot_approve_a_consent_setu_still_says_is_pending(env, monkeypatch):
    monkeypatch.setenv("SETU_WEBHOOK_SECRET", "s3cret")
    ask(env)
    r = webhook(env, {"consentId": "consent-1", "status": "ACTIVE", "data": {"status": "ACTIVE"}}, secret="s3cret")
    assert r.json() == {"status": "updated", "consent_status": "PENDING"}
    assert env["store"].get("sess-1").status == "PENDING"


def test_a_real_approval_arriving_by_webhook_updates_the_store(env, monkeypatch):
    monkeypatch.setenv("SETU_WEBHOOK_SECRET", "s3cret")
    ask(env)
    env["setu"].status = "ACTIVE"
    assert webhook(env, {"data": {"consentId": "consent-1"}}, secret="s3cret").json()["consent_status"] == "ACTIVE"
    assert env["store"].get("sess-1").status == "ACTIVE"


# ---------------------------------------------------------------- the existing credit path is untouched


def test_requests_without_an_aa_handle_still_use_the_original_credit_path(env):
    body = env["client"].post("/agent/run", json={"query": "can you check my credit score", "credit_signals": {"avg_monthly_income": 1}}).json()
    assert body["consent_status"] is None and body["credit_assessment"]["credit_score"] == 700
    assert env["setu"].created == 0
