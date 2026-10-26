"""API-level robustness tests with a stubbed router, retriever and LLM -- deterministic,
fast, no network and no model loading. They run the REAL compiled LangGraph and the real
FastAPI app; only the three expensive dependencies are replaced."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import agent.nodes.retrieve as retrieve_module
import agent.nodes.route as route_module
import api.main as main
import observability.tracing as tracing
from api import security
from api.response_cache import ResponseCache
from config import settings
from mlops.intent_router import drift_monitor
from observability.metrics import metrics
from tools.schemas import RetrievalResult, RetrievedChunk

DLG_CHUNK = RetrievedChunk(
    text=(
        "The total amount of DLG cover on any outstanding portfolio, specified upfront, shall not "
        "exceed five per cent of the total amount disbursed out of that loan portfolio at any given time."
    ),
    source="RBI_Digital_Lending_Directions_2025.pdf",
    page=7,
    score=0.99,
)
GOOD = json.dumps(
    {"answer": "The DLG cover cannot exceed 5% of the amount disbursed from the loan portfolio.",
     "confidence": 0.9, "escalate_to_human": False}
)
INVENTED = json.dumps(
    {"answer": "The cap is 5% of the amount disbursed, so for a Rs 40 crore portfolio it is Rs 2 crore.",
     "confidence": 0.9, "escalate_to_human": False}
)


class StubProvider:
    def __init__(self, behaviour):
        self.behaviour = behaviour  # str | Exception | list | callable
        self.calls: list[tuple[str, str]] = []
        self.last_usage = None

    def complete(self, system_prompt, user_query, max_tokens):
        self.calls.append((system_prompt, user_query))
        b = self.behaviour
        if callable(b):
            return b(self)
        if isinstance(b, list):
            b = b[min(len(self.calls) - 1, len(b) - 1)]
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_WARMUP", "1")
    monkeypatch.delenv("FINBUDDY_API_KEY", raising=False)
    monkeypatch.setattr(route_module, "classify_intent", lambda q: ("policy", 0.9))
    monkeypatch.setattr(
        retrieve_module,
        "retrieve_pdf_chunks",
        lambda query, store, top_k=5: RetrievalResult(chunks=[DLG_CHUNK], top_score=0.99, sufficient=True),
    )
    monkeypatch.setattr(tracing, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(settings, "FEEDBACK_PATH", tmp_path / "feedback" / "feedback.jsonl")
    monkeypatch.setattr(settings, "LLM_RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(security, "_limiter", security.RateLimiter(1000, 60))
    monkeypatch.setattr(main, "_cache", ResponseCache())
    metrics.reset()
    drift_monitor.reset()
    yield tmp_path
    drift_monitor.reset()


def use_provider(monkeypatch, behaviour) -> StubProvider:
    provider = StubProvider(behaviour)
    monkeypatch.setattr(main, "_default_provider", lambda: provider)
    return provider


def run(client, query="What is the cap on DLG cover?", **extra):
    return client.post("/agent/run", json={"query": query, **extra})


@pytest.fixture
def client():
    return TestClient(main.app)


# ---------------------------------------------------------------- failure handling


def test_provider_outage_escalates_cleanly_instead_of_500(client, monkeypatch):
    use_provider(monkeypatch, TimeoutError("read timed out"))
    r = run(client)
    assert r.status_code == 200
    body = r.json()
    assert body["escalate_to_human"] and body["failure_reason"] == "llm_unavailable"
    assert body["sources"] == [] and "human coach" in body["answer"]


def test_invalid_structured_output_after_retry_escalates(client, monkeypatch):
    use_provider(monkeypatch, "this is not json at all")
    body = run(client).json()
    assert body["escalate_to_human"] and body["failure_reason"] == "invalid_structured_output"


def test_an_answer_with_an_invented_figure_is_not_served(client, monkeypatch):
    use_provider(monkeypatch, INVENTED)
    body = run(client).json()
    assert body["escalate_to_human"] and body["failure_reason"] == "ungrounded_answer"
    assert "40" not in body["answer"] and body["sources"] == []
    assert "ungrounded_answer" in body["guardrail_flags"]


def test_a_grounded_answer_passes_and_cites_its_source(client, monkeypatch):
    use_provider(monkeypatch, GOOD)
    body = run(client).json()
    assert not body["escalate_to_human"] and body["failure_reason"] is None
    assert body["sources"] == ["RBI_Digital_Lending_Directions_2025.pdf"]


def test_an_unexpected_exception_in_the_graph_still_returns_a_clean_escalation(client, monkeypatch):
    use_provider(monkeypatch, GOOD)

    def boom(q, store, top_k=5):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(retrieve_module, "retrieve_pdf_chunks", boom)
    r = run(client)
    assert r.status_code == 200 and r.json()["failure_reason"] == "internal_error"


def test_one_transient_provider_error_is_retried_and_does_not_escalate(client, monkeypatch):
    provider = use_provider(monkeypatch, [ConnectionError("blip"), GOOD])
    body = run(client).json()
    assert body["failure_reason"] is None and not body["escalate_to_human"]
    assert len(provider.calls) == 2


# ---------------------------------------------------------------- privacy


def test_pii_never_reaches_the_llm_provider_or_the_trace_files(client, monkeypatch, isolated_app):
    provider = use_provider(monkeypatch, GOOD)
    body = run(client, "My PAN is ABCDE1234F and my number is 9876543210, what is the DLG cap?").json()
    assert sorted(body["pii_redacted"]) == ["mobile", "pan"]
    sent = " ".join(q for _, q in provider.calls)
    assert "ABCDE1234F" not in sent and "9876543210" not in sent
    traces = "".join(p.read_text() for p in (isolated_app / "traces").glob("*.jsonl"))
    assert traces and "ABCDE1234F" not in traces and "9876543210" not in traces


# ---------------------------------------------------------------- cache


def test_second_identical_policy_question_is_served_from_cache_without_an_llm_call(client, monkeypatch):
    provider = use_provider(monkeypatch, GOOD)
    first, second = run(client).json(), run(client, "  what is the CAP on dlg cover? ").json()
    assert not first["cache_hit"] and second["cache_hit"]
    assert len(provider.calls) == 1
    assert second["answer"] == first["answer"] and second["run_id"] != first["run_id"]


def test_failures_and_personal_requests_are_never_cached(client, monkeypatch):
    provider = use_provider(monkeypatch, [TimeoutError("x"), TimeoutError("y"), GOOD])
    assert run(client).json()["failure_reason"] == "llm_unavailable"  # 2 attempts, both fail
    assert run(client).json()["failure_reason"] is None  # not stuck on the failure
    assert len(provider.calls) == 3
    run(client, credit_signals={"avg_monthly_income": 1})
    assert len(main._cache) == 1  # only the one good policy answer


# ---------------------------------------------------------------- access control


def test_api_key_is_enforced_when_configured_and_open_otherwise(client, monkeypatch):
    use_provider(monkeypatch, GOOD)
    assert run(client).status_code == 200  # unset = open demo, as deployed today
    monkeypatch.setenv("FINBUDDY_API_KEY", "s3cret")
    assert run(client).status_code == 401
    assert client.post("/agent/run", json={"query": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/agent/run", json={"query": "What is the DLG cap?"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    assert client.get("/metrics").status_code == 401
    assert client.get("/health").status_code == 200  # health stays open for the platform probe


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    use_provider(monkeypatch, GOOD)
    monkeypatch.setattr(security, "_limiter", security.RateLimiter(2, 60))
    assert run(client).status_code == 200 and run(client).status_code == 200
    r = run(client)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


@pytest.mark.parametrize("bad", [{"session_id": "../../etc/passwd"}, {"query": ""}, {"query": "x" * 5000},
                                 {"response_language": "ta"}])
def test_invalid_requests_are_rejected_at_the_boundary(client, bad):
    payload = {"query": "What is the DLG cap?", **bad}
    assert client.post("/agent/run", json=payload).status_code == 422


# ---------------------------------------------------------------- monitoring + feedback


def test_metrics_reflect_traffic_failures_cache_and_tokens(client, monkeypatch):
    use_provider(monkeypatch, [GOOD])
    run(client)
    run(client)  # cache hit
    use_provider(monkeypatch, TimeoutError("down"))
    run(client, "A different question about DPDP consent?")
    snap = client.get("/metrics").json()
    assert snap["requests"] == 3 and snap["cache"]["hits"] == 1
    assert snap["fixed_fallbacks_by_reason"] == {"llm_unavailable": 1}
    assert snap["llm"]["calls"] == 1 and snap["llm"]["input_tokens"] > 0
    assert snap["latency_seconds"]["p50"] is not None


def test_feedback_roundtrip_updates_drift_and_stores_a_review_candidate(client, monkeypatch, isolated_app):
    use_provider(monkeypatch, GOOD)
    run_id = run(client, "Call 9876543210 about the DLG cap").json()["run_id"]
    r = client.post("/agent/feedback", json={"run_id": run_id, "rating": "down", "correct_route": "off_topic",
                                             "comment": "wrong, my PAN ABCDE1234F"})
    assert r.status_code == 200 and r.json()["route_was_correct"] is False
    assert drift_monitor.current_misroute_rate() == 1.0
    stored = [json.loads(line) for line in (isolated_app / "feedback" / "feedback.jsonl").read_text().splitlines()]
    assert stored[0]["correct_route"] == "off_topic" and stored[0]["reviewed"] is False
    assert "9876543210" not in json.dumps(stored) and "ABCDE1234F" not in json.dumps(stored)


def test_feedback_for_an_unknown_run_is_refused(client):
    r = client.post("/agent/feedback", json={"run_id": "does-not-exist", "rating": "up"})
    assert r.status_code == 404


# ---------------------------------------------------------------- streaming does not block the loop


def test_streaming_a_slow_run_does_not_freeze_other_requests(monkeypatch):
    def slow(provider):
        time.sleep(0.6)
        return GOOD

    use_provider(monkeypatch, slow)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            stream = asyncio.create_task(ac.post("/agent/run/stream", json={"query": "What is the DLG cap?"}))
            await asyncio.sleep(0.2)  # the stream is now inside the slow LLM call
            started = time.perf_counter()
            health = await ac.get("/health")
            health_seconds = time.perf_counter() - started
            await stream
            return health.status_code, health_seconds

    status, health_seconds = asyncio.run(scenario())
    assert status == 200
    assert health_seconds < 0.3, f"/health waited {health_seconds:.2f}s behind the stream (event loop blocked)"
