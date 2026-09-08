# FinBuddy Agentic RAG — LangGraph Build: Kickoff Prompt

Paste this whole file as the opening message to a fresh coding session (Claude Code
or otherwise) to start building. It is self-contained: goal, constraints, tech-stack
decisions, architecture, guardrails, evaluation, deployment, and a first milestone
checklist — plus how to talk about every piece of it afterward.

---

## Role

You are acting as an **expert AI Developer cum Product Manager**. Every architectural
choice below needs both an engineering justification (why this component, why this
library) and a product justification (why a user/judge should care) — write both when
you make a non-trivial choice, because they double as the demo script. The litmus test
for the whole build, borrowed directly from the curriculum: *"if you can't point to a
decision the model made on its own, you've built a wrapper, not an agent — say so
honestly."*

## What this project is

A **local-first, agentic RAG system** that lets FinBuddy's WhatsApp/PWA coach answer
questions grounded in **PDF documents** (RBI/DPDP policy PDFs, model cards, product
one-pagers) — built with **LangChain + LangGraph** this time, as a from-scratch
companion to the existing FinBuddy production build (`finbuddy-project/rag_service`),
which deliberately does NOT use LangChain (see that repo's `docs/curriculum_alignment.md`
— "retrofitting a LangChain layer just to match a tool name-check wasn't done"). This
project exists to make that agentic/LangGraph capability real, not to replace the
production repo. **Do not modify anything in the `finbuddy-project` repo from this
project** — this is a separate codebase in this folder (`IIT Kanpur/Langchain Langraph/`).

## Curriculum grounding (cite these when you present)

| Concept | Source | How it's used here |
|---|---|---|
| Agentic RAG pipeline: ingest → analyze → vector DB → query/retrieve/check/loop/generate/respond | Session 17 whiteboard | This *is* the LangGraph state machine below |
| RAG architectures: Basic → Multi-step → Recursive → **Agentic (LLM+Tools)** | Session 11 | We build the Agentic variant, not Basic RAG |
| Chunking (fixed-size vs semantic), dense vs sparse retrieval, re-ranking | Session 11 | Hybrid retrieval + re-rank step |
| 4 components of "agentic": LLM+role, memory, planning, tools | Session 12 | Mapped 1:1 to LangGraph node responsibilities below |
| Orchestration patterns: Parallel / Sequential / **Loop** / **Router** / Hierarchical / Network | Session 12 | Router node + Loop (retry-with-more-context) edge |
| Function calling: AI proposes JSON, code executes | Session 12 | Every tool call is validated before execution — a guardrail, not just a pattern |
| Structured prompt template (Role/Instructions/Constraints/Data/Output format) | Session 14, slide 31 | Every LLM call uses this exact structure |
| Token-cost reduction: query router, semantic cache, prompt cache, forced structured output, max_tokens, avoid hidden-reasoning models, summarize memory | Session 14, slide 42 | Directly implemented — see Cost & Latency section |
| Real deployment risks: compliance/legal, prompt injection, hallucination/brand, cost/energy | Session 14, slides 45–48 | Directly maps to the guardrails table below |
| 6-layer architecture (API/Agent/Tool/Memory/Guardrail/Observability), ReAct loop, LangGraph state machine, structured-output validation, evaluation harness, cost optimization, CI/CD gate | "Build Resume-Ready AI" blueprint (Roy's AI Lab) | This is the repo skeleton and the build-order checklist below |
| 7-layer GenAI stack (LLM / Framework / Vector DB / Data Extraction / Open LLM Access / Embeddings / Evaluation) with concrete tool options | LinkedIn stack post | Populates the Tech Stack Decisions table below |
| Decision framework: To Test/Validate → Statistics; To Predict/Forecast → Machine Learning; To Generate/Create → Generative AI | Session 18, slides 8–14 | The formal justification for "classical ML first, LLM only when mandatory" — the Intent Router is a Predict/classify problem (ML), not a Generate problem |
| "The Support Ticket Router" (route an email to Billing/Technical/Sales) and "The Quality Control Check" (is the machine drifting out of tolerance?) as worked decision scenarios | Session 18, slides 24–25 | Direct curriculum parallels to this project's Intent Router and its drift monitor — cite these by name in the demo, not just "we built a classifier" |
| "Treat AI as a decision-support tool, not an autonomous decision-maker"; "maintain a trail of AI assisted decisions" | Session 18, slide 39 | The credit-assessment path's actual approve/decline decision comes from production FinBuddy's governed API, never from this project's LLM — Generate only explains it. The trace log now records the actual answer text, not just a confidence number — see Guardrails table |
| "Wrong information — combine Gen AI with RAG and tools & add a disclaimer"; "Missed important information — generate guided summaries" | Session 18, slide 29 & 31 | Motivated the `disclaimer` field and the fixed (non-LLM) disclaimer text on every credit-assessment answer — see Guardrails table |

## Non-negotiable constraints (from the brief)

1. **Use AI/LLM/NLP only when a step genuinely requires language understanding or
   generation.** Everything else — routing, classification, ranking, sufficiency
   checks — must be classical ML or rules first. If you reach for an LLM call, write
   one sentence justifying why a classical method can't do it.
2. **Classical ML wherever possible.** Prefer scikit-learn, `rank_bm25`, regex/rules,
   cosine-similarity thresholds, logistic regression, etc. over an LLM call.
3. **Workflow laid out explicitly** — a LangGraph `StateGraph` with named nodes and
   conditional edges, not a single opaque chain.
4. **Minimal token usage** — every LLM call must be deliberate; see Cost & Latency
   Optimization below.
5. **LLM-agnostic** — the generation node calls an interface, not a hardcoded
   provider. Default to Claude (Anthropic API) for the demo; keep a local
   HuggingFace/Ollama adapter behind the same interface (this is the "why build
   agentic RAG" driver from the whiteboard: customization/ownership, cost control,
   private data, domain accuracy, local models).
6. **Each step must be independently explainable in a demo** — every node logs
   *what it decided and why* (not just its output), so you can screen-share the
   trace and narrate it live.
7. **Guardrails called out explicitly** — not implicit in code, but named,
   documented, and testable.

## Tech stack decisions (every row is a question an interviewer/judge will ask you to justify)

| Layer | Choice | Why it's defensible here |
|---|---|---|
| **LLM (the brain)** | Claude (Anthropic API), interface-agnostic | Reliable structured/function-calling output; swappable — see local-model row below |
| **Framework (the builder)** | LangChain + **LangGraph** (not LlamaIndex/Haystack) | Explicit state graph is inspectable and debuggable, not a black-box chain — the curriculum's own reason to prefer it over a plain RAG-only framework |
| **Vector DB (the memory)** | Chroma, local, session-scoped ("temporary storage" per the whiteboard) | Simple to self-host for a portfolio-scale project; `pgvector` is the noted alternative since production FinBuddy already uses it — not chosen here to keep this project dependency-light and local-first |
| **Data extraction (the parser)** | **Docling** (or LlamaParse) for PDF text/table/image extraction — not a web scraper (Crawl4AI/FireCrawl), since the source is PDFs, not websites | Matches Session 17's PDF-analysis requirement (text+metadata+chunking; tables→markdown+summary; images→detailed summary) directly |
| **Open LLM access (the free option)** | **Ollama** for the local-model fallback path; HuggingFace as the model hub it pulls from | Full local control, no per-token cost, keeps the "local AI models" driver from the whiteboard real rather than aspirational |
| **Text embeddings (the translator)** | Open-source local embedding model (sentence-transformers / NOMIC) as default; OpenAI/Voyage noted as a paid-quality alternative, not used by default | Keeps the pipeline runnable with zero external API cost for the retrieval path — only the Generate node spends paid tokens |
| **Evaluation (the tester)** | **RAGAS** (industry-standard RAG metrics: faithfulness, answer relevancy, context precision/recall) + a custom scenario harness for agent behavior (task completion, correct tool selection, step count) | RAGAS alone scores the *retrieval+generation* quality; it does NOT score whether the agent picked the right tool or looped efficiently — the custom harness (from the blueprint) covers that gap |
| **Serving** | FastAPI, async, streaming | Async-native, streams naturally to a frontend — turns a notebook script into a real, callable service |
| **Tracing** | LangSmith or OpenTelemetry | Gives a real, honest answer to "how do you debug it" instead of "I read the logs" |

## Architecture — six layers, top to bottom

```
1. API Layer        - FastAPI endpoint, auth, request validation, streaming
2. Agent Layer       - the ReAct loop below: planning, tool selection, state
3. Tool Layer        - retrieval, sufficiency-check, source-metadata — function-calling integrations
4. Memory Layer      - short-term (LangGraph state, per session) + long-term (none for MVP — see Memory below)
5. Guardrail Layer   - input validation, tool policy, output validation, structured-output enforcement
6. Observability     - tracing, evaluation hooks, cost/step logging
```

### The agent loop (ReAct, chosen deliberately over plan-execute)

```
Request → Agent Loop → Tools + Memory → Response
```

```python
# ReAct: reason, act, observe, repeat — chosen because query-answering over
# PDFs has real ambiguity in how much context is "enough"; plan-execute would
# force committing to a full plan up front, which fights the loop-back-for-
# more-context step the whiteboard specifies.
while not agent.goal_met():
    thought = llm.reason(state)          # only when genuinely needed — see constraint 1
    action  = agent.decide_next_step()   # classical router first, see table below
    result  = execute(action)
    agent.observe(result)
```

**Single-agent, not multi-agent** — deliberately. One planning loop with several tools
(retrieve, check-sufficiency, fetch-metadata) is simpler to debug and sufficient for
this scope; multi-agent (separate researcher/coder/reviewer agents) would only be
justified by real task specialization, which this problem doesn't have. State this
explicitly if asked — "building multi-agent because it sounds impressive" is a named
pitfall, not a strength.

### The LangGraph state machine (the agentic RAG loop itself)

```
PDF Ingestion
    ↓
PDF Analysis  (Docling: text+metadata+cleanup+chunking | tables→markdown+summary | images→detailed summary)
    ↓
Vector DB  (Chroma, local — "temporary storage" per the whiteboard, not a permanent
            system of record — a deliberate MVP scoping decision, not an oversight)
    ↓
┌─────────────── Agentic loop (LangGraph StateGraph) ───────────────┐
│  Query                                                             │
│    ↓                                                                │
│  Router (classical ML/rules — NOT an LLM call)                     │
│    ↓                                                                │
│  Retrieve (hybrid: dense embedding + BM25 sparse, then re-rank)    │
│    ↓                                                                │
│  Sufficiency Check (classical: top-hit CROSS-ENCODER score vs      │
│                      threshold — NOT raw cosine similarity, and    │
│                      NOT an LLM judging itself; see note below)    │
│    ↓ (insufficient) ──────► Add More Context ──► loop back to      │
│    ↓ (sufficient)                                    Retrieve      │
│  Generate (LLM call — the ONLY mandatory LLM step; structured      │
│             Role/Instructions/Constraints/Data/Output prompt,      │
│             Pydantic-validated JSON output, max_tokens set)        │
│    ↓                                                                │
│  Response  (+ cited sources, + confidence, + guardrail flags)      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Minimal LangGraph skeleton — draw this on a whiteboard before writing code,
# and be ready to redraw it in an interview; it's the strongest signal you
# built this rather than copy-pasted it.
graph = StateGraph(AgentState)
graph.add_node("route", classical_router_step)
graph.add_node("retrieve", hybrid_retrieve_step)
graph.add_node("check_sufficiency", sufficiency_check_step)
graph.add_node("generate", generate_step)
graph.add_conditional_edges("check_sufficiency", should_continue)  # loop vs proceed
```

Map to Session 12's four agent components:
- **LLM with defined role** → the Generate node's system prompt, scoped narrowly
  (FinBuddy coach persona, same tone rules as production `finbuddy_system_prompt.txt`).
- **Memory** → short-term: LangGraph state carried across the loop, cleared per
  session; long-term: none needed for MVP — call this out explicitly as a scoping
  decision, don't half-build it. If added later: retrieve top-k relevant memories
  by search, don't stuff full history into every prompt (burns tokens, buries
  what's relevant).
- **Planning** → the Router + Sufficiency-Check nodes *are* the planning layer, done
  classically instead of via LLM chain-of-thought (the "AI/NLP only when mandatory"
  principle in action — name it as such in the demo).
- **Tools** → `retrieve_pdf_chunks`, `check_sufficiency`, `fetch_source_metadata` —
  each exposed as a typed schema (name, description, typed parameters) the model
  reasons over. Write tool descriptions the way you'd explain the tool to a new
  teammate — vague descriptions are the top cause of the model calling the wrong
  tool (a named, common bug — see Debugging Scenario below).

## Where classical ML replaces an LLM call (be ready to defend each in the demo)

| Step | Classical approach | Why not an LLM here |
|---|---|---|
| Query routing (policy question vs coaching question vs off-topic) | `scikit-learn` text classifier (TF-IDF + logistic regression) or keyword rules | Deterministic, cheap, and the intent classifier in production FinBuddy already proves LaBSE-embedding classifiers beat LLM-based routing on cost and latency |
| Sparse retrieval | `rank_bm25` | Classical IR, no embedding cost |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, sentence-transformers) — not an LLM prompt | Ranking is a scoring problem, not a generation problem. **Load-bearing, not cosmetic**: verified on real ingested PDFs that raw bi-encoder cosine similarity topped out at ~0.45 for genuinely correct matches on this dense legal-text corpus, while the cross-encoder scored the same true match at 0.996 (post-sigmoid) and an irrelevant chunk at ~0.00003 — a bi-encoder's cosine score is a ranking signal, not a calibrated confidence signal, and doesn't transfer across corpora with different register/length |
| Sufficiency check | Threshold (0.5) on the cross-encoder's sigmoid-mapped score — NOT on raw cosine similarity | Deterministic and auditable, and — after the re-ranking fix above — actually calibrated against this corpus's real score distribution, not an inherited number that turned out to make the RAG path unreachable (every real query escalated under the old cosine threshold before this fix) |
| PII/compliance flag on input | Regex + a small classical classifier | Don't send unredacted PII to an LLM at all |
| Route-by-difficulty (cost optimization) | Rule/classical check on query length/complexity | Simple queries shouldn't invoke the full agent loop at all — only complex ones need it |

## Structured output & validation (reliability)

Free-text agent output is the hardest thing to build a reliable system on. Force the
Generate node to return a **Pydantic schema**, not prose:

```python
class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    escalate_to_human: bool
```

On a validation failure, **retry once with the error message fed back to the model** —
don't crash on a bad parse. This turns "parse the model's prose" into "validate a
structured object," and it's the same discipline production FinBuddy's JSON-only
output contract already uses.

## Guardrails (name these explicitly in the submission)

Guardrails validate input before it reaches the model, constrain which tools can run
in which context, and check output before it's returned or acted on — catching what
the prompt alone won't.

| Guardrail | What it prevents | How enforced |
|---|---|---|
| Grounding gate | Hallucinated policy/compliance answers | Sufficiency-check node blocks Generate before the LLM ever runs; **and**, as defense-in-depth, `generate_node` itself now overrides `escalate_to_human` to `True` whenever `sufficient is False`, regardless of what the model self-reports — verified necessary, not theoretical: the HF fallback model answered confidently instead of escalating on one real run and escalated correctly on another, purely by chance, on an identical input |
| Structured, forced-JSON output | Prompt injection via free-form output, downstream parsing bugs | `Generate` node requests strict Pydantic schema; reject and retry once on schema violation, error fed back to the model |
| Input sanitization | Prompt injection from PDF content or user query (Session 14's Bing-jailbreak case study) | Strip/escape control sequences from retrieved chunks before they enter the prompt; never let retrieved text carry instructions the LLM will obey |
| Confirmation gate on side-effecting tools | An agent acting on bad judgment with real consequences | **Non-negotiable**: any tool with a real-world side effect (sending a message, writing to a DB) needs an explicit confirmation gate — this MVP is read-only (retrieval only), so document this as "not yet triggered" rather than skip it silently if a write-capable tool is added later |
| Token/cost ceiling | Runaway spend, the loop edge looping forever | Hard cap on loop iterations (`max_steps`, e.g. 2–3) and `max_tokens` per call; log every LLM call's token count |
| No protected-attribute / PII leakage | DPDP/compliance violation | Same rule as production FinBuddy's system prompt — never use gender/religion/caste/pincode as signals, never echo PII back |
| Human-in-the-loop escalation | Silent wrong answers on out-of-scope or low-confidence queries | Router/Sufficiency-Check can force a "route to human" terminal node, not just the Generate node |
| Local-model fallback path | Sending sensitive data to a third-party API when not appropriate | LLM interface is provider-agnostic; document when you'd flip to the local Ollama model |
| Tool-failure escalation on the credit-assessment path | An LLM narrating a failed tool call's fallback data (e.g. `credit_score=0`) as if it were a real result | **Found as a real bug in a Session 18 guardrail review**: `credit_assessment_node` used to mark state "sufficient" and flow straight to Generate even when `assess_credit_profile` failed. Fixed with a conditional edge in `agent/graph.py` — a `tool_error` routes to escalate, never to Generate. Regression-tested in `tests/test_agent_graph.py` |
| Decision-support disclaimer | A user mistaking an AI-generated explanation for financial advice or an autonomous decision | Session 18's "AI as decision-support, not autonomous decision-maker": the actual approve/decline decision always comes from production FinBuddy's governed, audited API — Generate only explains it, and every credit-assessment answer carries a fixed, code-appended (never LLM-generated) disclaimer — see `agent/nodes/generate.py`'s `CREDIT_ASSESSMENT_DISCLAIMER` |
| Audit trail of AI-assisted decisions | "What did the AI actually tell this user?" being unanswerable after the fact | Session 18, slide 39. `observability/tracing.py`'s per-run trace now logs the Generate node's actual answer text, not just a confidence score — a confidence number alone doesn't let anyone reconstruct what was said |
| Explicit tool-error handling | A flaky tool call crashing the whole run | Catch tool errors explicitly, feed the error back to the agent as an *observation*, not a crash; cap retries per tool call and per overall run; define a fallback (degrade to a simpler answer or escalate) rather than fail silently |

## Cost & Latency Optimization

Every extra loop iteration is an extra LLM call on the bill. Implement all of these,
not just the ones that are convenient:

- [ ] Query router runs before any LLM call (classical, see table above)
- [ ] **Cap iterations** — a hard `max_steps` stops runaway loops on ambiguous queries
- [ ] **Cache tool/retrieval results** — repeated identical calls within a run shouldn't
      hit the vector DB or embedding model twice
- [ ] Semantic cache: hash+embed repeated *queries*, skip regeneration on a cache hit
- [ ] Prompt caching: static system prompt + document context cached across calls
      (Anthropic prompt caching, since Claude is the default LLM)
- [ ] **Route by difficulty** — simple queries skip the full agent loop entirely
      (classical check); only ambiguous ones invoke retrieval+generation
- [ ] **Parallelize independent tool calls** where the graph allows it, instead of
      running them sequentially
- [ ] Forced structured JSON output (no wasted tokens on prose)
- [ ] Explicit `max_tokens` set on every call, input and output
- [ ] Use a non-hidden-reasoning model for the demo (avoid extended-thinking token
      overhead when it's not needed for this task)
- [ ] Summarize conversational memory instead of replaying full history
- [ ] **Know your numbers** — track and be ready to quote average steps-per-run and
      cost-per-run; this is a specific, interviewer-tested detail, not a nice-to-have

## Serving & Observability

- **Serving**: FastAPI async endpoint (`POST /agent/run`) that invokes the graph.
  **Stream** intermediate steps ("Calling retriever...") and final tokens as they
  happen — perceived latency matters as much as actual latency, and users tolerate
  a multi-step agent far better when they can see it thinking.
- **Tracing**: wrap every run with a tracer (LangSmith or OpenTelemetry), keyed by a
  session/run id, so a bad output can be traced back to the exact decision that
  caused it — not guessed at. Trace **tool selection specifically** (the model's
  stated reason for picking a tool, not just which one it picked) — this is what
  separates a fixable bug from a mystery.

## Evaluation harness

You can't improve decisions you never measured. Build a fixed set of scenarios with
expected tool calls and outcomes, run on every change:

```python
result = run_agent(scenario.query)
task_completed = check_goal(result, scenario.expected)
right_tools = result.tools_used == scenario.expected_tools
```

Score **task completion, correct tool selection, and number of steps taken** — not
just final-answer text. **Split the blame**: a wrong final answer might be a wrong
tool choice, a bad retrieval result, or a good tool call with bad synthesis — score
each separately so you can say exactly which layer failed. Layer RAGAS metrics
(faithfulness, answer relevancy, context precision/recall) on top for the
retrieval+generation quality specifically.

Wire this into CI: a GitHub Actions step that runs the eval harness on every PR and
blocks the merge on a regression — this shows you treat agent behavior as testable
software, not a lucky prompt.

```yaml
on: pull_request
jobs:
  eval:
    steps:
      - run: pytest tests/
      - run: python evaluate_agent.py --gate
```

## Deployment

A GitHub repo isn't a deployed application. Containerize and deploy behind a real
endpoint — "deployed on [cloud service] with a live demo link" is worth more than
any amount of local notebook polish:

```dockerfile
FROM python:3.11-slim
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

`Dockerfile → Image → Cloud Run/ECS` (or the equivalent free-tier target you already
use for the production FinBuddy Spaces deployment, for consistency).

## Common pitfalls — self-check before the demo

Walk through this list against the actual build; each unaddressed gap is a follow-up
question waiting to happen:

- [ ] **No iteration cap** — an agent loop without `max_steps` can retry indefinitely
- [ ] **Vague tool descriptions** — the top cause of the model calling the wrong tool
- [ ] **No confirmation gate** on any tool with a real side effect
- [ ] **Only the happy path tested** — never testing tool failures means the first
      real hiccup breaks the whole run

## Debugging scenario worth rehearsing

*"Your agent has two similar tools — `retrieve_pdf_chunks` and (if added later)
`retrieve_policy_summaries` — and it keeps picking the wrong one for a query that
should use the other. Both work fine in isolation."* The bug isn't in either
component alone — it's in how the model is choosing between them. Fix order: (1)
check the tool descriptions for near-identical wording, (2) check few-shot examples
in the system prompt for an anchoring bias, (3) add tracing on tool selection
specifically (log the *stated reason*, not just the chosen tool), (4) sharpen
descriptions / balance examples, then re-run the eval harness to confirm the fix
generalizes, not just patches the one case you saw.

## Submission shape (reuse the proven AIPL capstone format from `FinBuddy_MLOps_Capstone.pptx`)

1. **Why this matters** — one live statistical/product claim, kept honest.
2. **Mandatory questions** (however this phase's brief phrases them) — answer with
   real evidence, not projected numbers.
3. **One featured deep-dive** — pick ONE node (likely the Router or the
   Sufficiency-Check) and show every configuration tested, not just the final one —
   mirrors the risk-trend-classifier deep-dive's credibility.
4. **Responsible AI / guardrails gate table** — reuse the table above directly.
5. **Strategic value close** — why agentic + local-model-optional beats a single
   hardcoded Groq call for this use case (ownership, cost, domain accuracy — same
   four reasons as the Session 17 whiteboard's "why build agentic RAG").

## Telling the story (STAR structure, for an interview or a live demo walkthrough)

- **Situation**: the problem, and why a single LLM call couldn't solve it (FinBuddy
  needs grounded, auditable answers over PDFs it wasn't trained on).
- **Task**: what the agent needed to decide and do autonomously (route, retrieve,
  judge sufficiency, loop or answer).
- **Action**: the architecture choices — agent pattern (ReAct, single-agent), tools,
  memory scope, guardrails.
- **Result**: what you measured — task success rate, latency, cost per run. A
  project with no measured outcome reads as unfinished, no matter how sophisticated
  the architecture — don't skip this.

Resume-line template: *"Built and deployed a [domain] agent using [orchestration
framework] with [N] tools, [memory type], and a guardrail layer — reducing [metric]
by [X%] / handling [Y] scenario types."* Fill this in for real once the eval numbers
exist — don't leave it as a placeholder in the final submission.

## First milestone — what to build in the first session

1. Scaffold the repo along the six layers: `api/`, `agent/` (LangGraph nodes + state
   schema), `tools/`, `memory/`, `guardrails/`, `observability/`, plus `ingestion/`
   and `eval/`.
2. Pick 3–5 real PDFs to start with (reuse `finbuddy-project/rag_service/finbuddy_rag_corpus/`
   source *content*, converted to PDF, or the RBI/DPDP source PDFs if available —
   don't fabricate policy text).
3. Build PDF ingestion + analysis (Docling: text/tables/images) → local Chroma vector DB.
4. Build the classical router + BM25/dense hybrid retrieval + sufficiency check —
   get this working and logging its decisions *before* wiring in any LLM call.
5. Wire the LangGraph state machine end-to-end with a stub Generate node (echo
   retrieved context) to prove the loop/router/edges work. Draw the graph before
   writing this step's code.
6. Swap in the real Generate node (Claude API, structured Role/Instructions/
   Constraints/Data/Output prompt, Pydantic-validated JSON, max_tokens set).
7. Add the guardrail tests (grounding gate, injection resistance, token-cap
   enforcement, tool-error handling) as an actual test suite — same rigor as
   production FinBuddy's Gate A/Gate B tests.
8. Add the evaluation harness (scenario set + RAGAS) and wire it into a CI gate.
9. Add tracing (LangSmith/OpenTelemetry) and streaming on the FastAPI endpoint.
10. Containerize and deploy; capture a live demo link, not just a notebook.

## Decisions (resolved — see `build_prompt.md` for the full build spec)

- **Demo corpus**: real RBI/DPDP source PDFs, not the converted markdown corpus —
  don't fabricate or substitute policy text.
- **Local-Ollama fallback**: verify it works at least once outside the live demo (a
  real, run-and-confirmed test), but don't run it live in the demo path — same
  "real but not always exercised" honesty pattern as production FinBuddy's README
  status table. The story ("local model option exists for cost/privacy") is proven
  by the interface being real and tested, not by live-switching providers on stage.
- **Deployment**: HuggingFace Spaces, as a **new, separate Space** — do not touch or
  redeploy the existing production FinBuddy Spaces or repo.

`build_prompt.md` in this same folder is the actionable next document: it resolves
these with full detail, adds the classical-ML algorithm selections (reusing
production FinBuddy's proven models) and the MLOps loop, and gives an updated
milestone checklist. Read this file first for the "why," then `build_prompt.md`
for the "what to build, concretely."
