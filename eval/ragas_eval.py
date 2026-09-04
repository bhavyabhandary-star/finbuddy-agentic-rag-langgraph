"""RAGAS metrics (faithfulness, answer relevancy, context precision/recall) for
the retrieval+generation quality specifically — complements eval/evaluate_agent.py,
which scores agent-level behavior (routing, tool selection) that RAGAS doesn't
touch. See kickoff_prompt.md's Tech Stack Decisions Evaluation row.

TODO(milestone step 8): wire real (question, retrieved_contexts, answer,
ground_truth) tuples once the real RBI/DPDP PDFs are ingested — RAGAS needs real
retrieval output to score meaningfully, so this is a stub until then.
"""
from __future__ import annotations


def evaluate_rag_quality(qa_pairs: list[dict]) -> dict:
    """qa_pairs: list of {"question", "contexts", "answer", "ground_truth"}.

    Returns RAGAS's faithfulness / answer_relevancy / context_precision /
    context_recall scores, averaged across qa_pairs.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        raise RuntimeError("ragas is not installed — run `pip install -r requirements.txt`.")

    dataset = Dataset.from_list(qa_pairs)
    result = evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    return result
