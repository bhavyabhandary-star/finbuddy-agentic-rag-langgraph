"""Independent-judge answer-quality eval: a RAGAS-free fallback, and a check on the RAGAS numbers.

This is NOT RAGAS. It exists because RAGAS cannot be imported on every machine (on this
project's Windows dev box its native dependency, tiktoken, is blocked by an Application Control
policy), and because the earlier RAGAS runs were judged by the same Qwen model that wrote the
answers. Three measures, each labelled for what it is:

  claim_faithfulness  an LLM JUDGE (a different model family from the generator) splits the answer
                      into claims and marks each supported / not supported by the retrieved text
  correctness         the same judge: does the answer state the ground-truth fact without
                      contradicting it?
  context_recall      DETERMINISTIC: share of the ground truth's content words found in the
                      retrieved text (no model involved)

plus the runtime grounding check (guardrails/grounding.py), also deterministic.

An 8B judge is a noisy grader: treat the scores as a regression signal and a sanity check, not as
a certificate. n is small; every per-question row is written out so a human can read them.

Run: python -m eval.judge_eval [--out docs/judge_eval_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date

from config import settings
from eval.ragas_eval import REAL_QA_SEEDS, build_real_qa_pairs
from guardrails.grounding import _content_words, check_grounding

_FAITHFULNESS_PROMPT = """You are a strict fact-checker. CONTEXT is the only source of truth.
Split the ANSWER into its distinct factual claims. For each claim say whether CONTEXT supports it.
A claim that adds a detail CONTEXT does not state is NOT supported.

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true}}]}}"""

_CORRECTNESS_PROMPT = """You are grading an answer against a reference fact.
Does the ANSWER state the same essential fact as the REFERENCE, without contradicting it?
Extra correct detail is fine. A missing or wrong key fact (a number, a duration, a condition) is not.

REFERENCE:
{reference}

ANSWER:
{answer}

Return ONLY JSON: {{"correct": true, "reason": "one short sentence"}}"""


def _judge_json(client, model: str, prompt: str) -> dict:
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    truth_words = _content_words(ground_truth)
    if not truth_words:
        return 1.0
    return len(truth_words & _content_words(" ".join(contexts))) / len(truth_words)


def evaluate(pairs: list[dict]) -> dict:
    from huggingface_hub import InferenceClient

    judge = settings.ragas_judge_model()
    generator = settings.hf_model()
    client = InferenceClient(provider=settings.hf_provider(), token=settings.hf_token(), timeout=settings.LLM_TIMEOUT_SECONDS)

    rows = []
    for p in pairs:
        context = "\n---\n".join(p["contexts"])
        row = {"question": p["question"], "answer": p["answer"], "escalated": p.get("escalated", False)}
        if row["escalated"]:
            rows.append(row)
            continue
        try:
            claims = _judge_json(client, judge, _FAITHFULNESS_PROMPT.format(context=context, answer=p["answer"]))["claims"]
            supported = sum(bool(c.get("supported")) for c in claims)
            row["claim_faithfulness"] = supported / len(claims) if claims else None
            row["unsupported_claims"] = [c.get("claim") for c in claims if not c.get("supported")]
            verdict = _judge_json(client, judge, _CORRECTNESS_PROMPT.format(reference=p["ground_truth"], answer=p["answer"]))
            row["correct"] = bool(verdict.get("correct"))
            row["correctness_reason"] = verdict.get("reason")
        except Exception as exc:  # noqa: BLE001 -- one bad judge reply must not sink the whole run
            row["judge_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["context_recall"] = round(context_recall(p["ground_truth"], p["contexts"]), 3)
        grounding = check_grounding(p["answer"], p["contexts"], p["question"])
        row["runtime_grounding_passed"] = grounding.grounded
        rows.append(row)

    scored = [r for r in rows if "claim_faithfulness" in r and r["claim_faithfulness"] is not None]
    correctness = [r for r in rows if "correct" in r]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "run_at": date.today().isoformat(),
        "generator": generator,
        "judge": judge,
        "judge_is_generator": judge == generator,
        "n_questions": len(pairs),
        "n_answered": sum(not r["escalated"] for r in rows),
        "n_escalated": sum(r["escalated"] for r in rows),
        "n_judge_errors": sum("judge_error" in r for r in rows),
        "claim_faithfulness_mean": mean([r["claim_faithfulness"] for r in scored]),
        "correct": f"{sum(r['correct'] for r in correctness)}/{len(correctness)}",
        "context_recall_mean": mean([r["context_recall"] for r in rows if "context_recall" in r]),
        "runtime_grounding_pass_rate": mean([float(r["runtime_grounding_passed"]) for r in rows if "runtime_grounding_passed" in r]),
        "rows": rows,
    }


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pairs = build_real_qa_pairs()
    report = evaluate(pairs)
    for r in report["rows"]:
        flag = "ESCALATED" if r["escalated"] else f"faith={r.get('claim_faithfulness')} correct={r.get('correct')} recall={r.get('context_recall')} grounded={r.get('runtime_grounding_passed')}"
        print(f"- {r['question']}\n    {flag}")
        for claim in r.get("unsupported_claims", []):
            print(f"    unsupported claim: {claim}")
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
