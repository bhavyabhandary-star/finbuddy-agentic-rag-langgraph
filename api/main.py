"""API Layer (Layer 1): FastAPI endpoint, request validation, streaming.

Run: uvicorn api.main:app --reload --port 8010

Beyond routing requests into the graph, this layer is where the cross-cutting robustness
controls live (each one documented in docs/robustness_checklist.md):
  access control  X-API-Key (opt-in) + per-client rate limit          api/security.py
  privacy         PII redacted BEFORE the router / LLM / trace        guardrails/input_guardrails.py
  failure         a failed graph run becomes a clean escalation, never a 500
  cost / latency  TTL cache for grounded policy answers; model + index warmed at startup
  monitoring      GET /metrics (volume, escalations, p50/p95, tokens, router drift)
  feedback        POST /agent/feedback -> reviewed corrections -> retraining
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import iterate_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.graph import _DEFAULT_ESCALATION_MESSAGE, build_graph
from agent.nodes.generate import (
    ClaudeProvider,
    HuggingFaceProvider,
    LLMProvider,
    OllamaProvider,
)
from api.feedback import FeedbackRequest, RecentRun, RecentRuns, append_feedback, build_record
from api.response_cache import ResponseCache
from api.security import enforce_rate_limit, require_api_key
from config import settings
from guardrails.input_guardrails import redact_pii
from ingestion.setu_feed import load_cached_real_profile
from mlops.intent_router import drift_monitor
from observability.metrics import metrics
from observability.tracing import trace_run
from tools.schemas import CreditAssessmentResult

load_dotenv()

logger = logging.getLogger("api")


def warm_up() -> None:
    """Load everything the first request would otherwise load lazily.

    Measured on the deployed Space: the first policy query took ~20s (cross-encoder weights
    and the Chroma index loading inside the request) against ~5s once warm. Paying that at
    startup moves it out of the first user's wait. SKIP_WARMUP=1 skips it (tests).
    """
    from agent.nodes.retrieve import _vector_store
    from mlops.intent_router.infer import classify_intent
    from tools.rag_tools import retrieve_pdf_chunks

    classify_intent("warm up")
    retrieve_pdf_chunks("warm up", _vector_store, top_k=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("SKIP_WARMUP") != "1":
        try:
            warm_up()
        except Exception:  # noqa: BLE001 -- a warm-up failure must not stop the API from starting
            logger.exception("warm_up_failed")
    yield


app = FastAPI(title="FinBuddy Agentic RAG (LangGraph)", lifespan=lifespan)

# Public read-only demo API, no cookies/session auth to protect -- open CORS
# is the right tradeoff here (a browser-hosted demo UI needs cross-origin
# fetch access, and there's nothing origin-based auth would otherwise gate).
# Access control is the X-API-Key + rate limit in api/security.py, not CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_cache = ResponseCache()
_recent_runs = RecentRuns()
_PROTECTED = [Depends(require_api_key), Depends(enforce_rate_limit)]


def _default_provider() -> LLMProvider:
    """Env-configurable provider selection (LLM_PROVIDER=claude|huggingface|
    ollama, defaults to claude) — kickoff_prompt.md's LLM-agnostic design
    goal, made real: swapping providers for the deployed API needs an env
    var, not a code change. build_graph()'s injectable `provider` param
    (added for testing while Claude billing was blocked) is what makes this
    possible without touching agent/graph.py at all.
    """
    choice = settings.llm_provider_name()
    if choice == "huggingface":
        return HuggingFaceProvider()
    if choice == "ollama":
        return OllamaProvider()
    return ClaudeProvider()


_UPI_SIGNAL_KEYS = (
    "avg_monthly_income",
    "income_regularity_score",
    "tx_count_30d",
    "merchant_diversity",
    "balance_dip_frequency",
    "b2b_ratio",
    "avg_transaction_size",
    "tenure_months",
)


class AgentRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.MAX_QUERY_CHARS)
    # Client-chosen conversation id. Only used as a label: the trace filename is a fresh
    # per-request id, so a hostile value can never become a path.
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # "en" (default) | "hi" | "kn" -- only the Generate node's answer text and
    # the deterministic escalation/disclaimer strings honor this; routing and
    # retrieval always use the original query, see agent/state.py's note.
    response_language: Literal["en", "hi", "kn"] = "en"
    # Present only for a credit-assessment request — the API layer, not the
    # agent, is responsible for collecting these (see agent/state.py's note).
    # risk_trend_delta_features must already be z-scored against the training
    # population — see tools/credit_tools.py's assess_risk_trend docstring.
    credit_signals: dict | None = None
    risk_trend_delta_features: dict | None = None
    # When true, ignores credit_signals and uses the real, most-recently
    # pulled Setu AA sandbox profile instead — the capstone's own "Setu AA
    # Feed" pipeline stage, see ingestion/setu_feed.py. Real sandbox data,
    # not synthesized: consent-based 12-mo UPI pull, normalized to 8 signals.
    use_setu_feed: bool = False


class AgentRunResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    escalate_to_human: bool
    disclaimer: str | None = None
    # Additive decision-trace fields, all optional so existing callers (the
    # verified curl demo commands) are unaffected -- added for a UI that
    # shows the real routing/retrieval decision, not just the final answer.
    route: str | None = None
    route_confidence: float | None = None
    top_score: float | None = None
    sufficient: bool | None = None
    loop_count: int | None = None
    # Structured F-001/F-003/F-006/F-012 score data for a credit_assessment
    # request -- the "answer" field is only the LLM's plain-English narration
    # of this same data; a UI wanting a score hero / factor bars needs the
    # numbers themselves, not a re-parse of prose.
    credit_assessment: CreditAssessmentResult | None = None
    # Robustness fields (all additive). run_id is what POST /agent/feedback refers to.
    run_id: str | None = None
    failure_reason: str | None = None  # why the answer is a fixed fallback, if it is
    guardrail_flags: list[str] = []
    pii_redacted: list[str] = []  # kinds removed from the query before it was processed
    cache_hit: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def get_metrics() -> dict:
    snapshot = metrics.snapshot()
    snapshot["cache"] = {"entries": len(_cache), "hits": _cache.hits, "misses": _cache.misses}
    return snapshot


def _resolve_credit_signals(request: "AgentRunRequest") -> dict:
    if not request.use_setu_feed:
        return request.credit_signals or {}
    profile = load_cached_real_profile()
    if profile is None:
        return {}  # no real profile pulled yet — assess_credit_profile's own
        # guardrail (tool_error on a bad/incomplete request) covers this, not
        # a fabricated fallback here.
    return {k: profile[k] for k in _UPI_SIGNAL_KEYS if k in profile}


def _cacheable_request(request: AgentRunRequest) -> bool:
    """A policy answer depends only on (question, language). Anything carrying the caller's
    own credit data is personal and is never served from, or stored in, the cache."""
    return not (request.use_setu_feed or request.credit_signals or request.risk_trend_delta_features)


def _initial_state(request: AgentRunRequest, session_id: str, query: str) -> dict:
    return {
        "query": query,
        "session_id": session_id,
        "response_language": request.response_language,
        "credit_signals": _resolve_credit_signals(request),
        "risk_trend_delta_features": request.risk_trend_delta_features,
    }


def _internal_error_response(request: AgentRunRequest, run_id: str, pii_kinds: list[str]) -> AgentRunResponse:
    return AgentRunResponse(
        answer=_DEFAULT_ESCALATION_MESSAGE.get(request.response_language, _DEFAULT_ESCALATION_MESSAGE["en"]),
        sources=[],
        confidence=0.0,
        escalate_to_human=True,
        run_id=run_id,
        failure_reason="internal_error",
        pii_redacted=pii_kinds,
    )


def _response_from_result(result: dict, run_id: str, pii_kinds: list[str]) -> AgentRunResponse:
    return AgentRunResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        escalate_to_human=result.get("escalate_to_human", False),
        disclaimer=result.get("disclaimer"),
        route=result.get("route"),
        route_confidence=result.get("route_confidence"),
        top_score=result.get("top_score"),
        sufficient=result.get("sufficient"),
        loop_count=result.get("loop_count"),
        credit_assessment=result.get("credit_assessment"),
        run_id=run_id,
        failure_reason=result.get("failure_reason"),
        guardrail_flags=result.get("guardrail_flags", []),
        pii_redacted=pii_kinds,
    )


def _finish(
    response: AgentRunResponse, request: AgentRunRequest, query: str, started: float, cost_summary: dict | None
) -> AgentRunResponse:
    _recent_runs.remember(
        response.run_id or "", RecentRun(query=query, route=response.route, escalated=response.escalate_to_human)
    )
    metrics.record_request(
        route=response.route,
        route_confidence=response.route_confidence,
        escalated=response.escalate_to_human,
        failure_reason=response.failure_reason,
        cache_hit=response.cache_hit,
        latency_seconds=time.perf_counter() - started,
        pii_redacted=bool(response.pii_redacted),
        cost_summary=cost_summary,
    )
    return response


@app.post("/agent/run", response_model=AgentRunResponse, dependencies=_PROTECTED)
def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    started = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    query, pii_kinds = redact_pii(request.query)

    cacheable = _cacheable_request(request)
    if cacheable:
        cached = _cache.get(query, request.response_language)
        if cached is not None:
            with trace_run(run_id=run_id) as tracer:
                tracer.log_step("api", decision="served from response cache", session_id=session_id, query=query)
            hit = cached.model_copy(update={"run_id": run_id, "cache_hit": True, "pii_redacted": pii_kinds})
            return _finish(hit, request, query, started, None)

    cost_summary = None
    with trace_run(run_id=run_id) as tracer:
        tracer.log_step("input_guardrail", decision="query screened", session_id=session_id, pii_redacted=pii_kinds)
        try:
            graph = build_graph(tracer=tracer, provider=_default_provider())
            result = graph.invoke(_initial_state(request, session_id, query))
            response = _response_from_result(result, run_id, pii_kinds)
        except Exception:  # noqa: BLE001 -- fail closed: a broken run escalates, it does not 500
            logger.exception("agent_run_failed", extra={"run_id": run_id})
            tracer.log_step("api", decision="graph run raised; returned a fixed escalation")
            response = _internal_error_response(request, run_id, pii_kinds)
        cost_summary = tracer.cost.summary()

    if cacheable and response.route == "policy" and not response.escalate_to_human and not response.failure_reason:
        _cache.put(query, request.response_language, response)
    return _finish(response, request, query, started, cost_summary)


@app.post("/agent/run/stream", dependencies=_PROTECTED)
async def run_agent_stream(request: AgentRunRequest):
    """Streams intermediate step names before the final answer — perceived
    latency matters as much as actual latency (kickoff_prompt.md's Serving
    section: "surface which tool is running, not just final tokens").

    The graph is synchronous and blocks on network calls, so it is consumed through
    iterate_in_threadpool: iterating it directly inside this async generator froze the
    event loop, stalling every other request on the worker for the length of the run.
    """
    started = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    query, pii_kinds = redact_pii(request.query)

    async def event_generator():
        final_state = None
        with trace_run(run_id=run_id) as tracer:
            tracer.log_step("input_guardrail", decision="query screened", session_id=session_id, pii_redacted=pii_kinds)
            try:
                graph = build_graph(tracer=tracer, provider=_default_provider())
                async for step_output in iterate_in_threadpool(graph.stream(_initial_state(request, session_id, query))):
                    node_name = next(iter(step_output))
                    yield {"event": "step", "data": node_name}
                    final_state = step_output[node_name]
            except Exception:  # noqa: BLE001
                logger.exception("agent_stream_failed", extra={"run_id": run_id})
                final_state = {"failure_reason": "internal_error", "escalate_to_human": True,
                               "answer": _DEFAULT_ESCALATION_MESSAGE.get(request.response_language, _DEFAULT_ESCALATION_MESSAGE["en"])}
            cost_summary = tracer.cost.summary()
        if final_state:
            yield {"event": "final", "data": final_state.get("answer", "")}
            _finish(_response_from_result(final_state, run_id, pii_kinds), request, query, started, cost_summary)

    return EventSourceResponse(event_generator())


@app.post("/agent/feedback", dependencies=_PROTECTED)
def submit_feedback(request: FeedbackRequest) -> dict:
    run = _recent_runs.get(request.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown or expired run_id.")

    route_was_correct = request.route_was_correct
    if route_was_correct is None and request.correct_route is not None and run.route is not None:
        route_was_correct = request.correct_route == run.route
    if route_was_correct is not None:
        drift_monitor.record_routing_outcome(route_was_correct)

    append_feedback(build_record(request, run, route_was_correct))
    return {"status": "recorded", "route_was_correct": route_was_correct}
