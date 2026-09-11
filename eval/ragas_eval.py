"""RAGAS metrics (faithfulness, answer relevancy, context precision/recall) for
the retrieval+generation quality specifically — complements eval/evaluate_agent.py,
which scores agent-level behavior (routing, tool selection) that RAGAS doesn't
touch. See kickoff_prompt.md's Tech Stack Decisions Evaluation row.

Judge LLM and embeddings default to non-OpenAI: RAGAS's stock defaults require
OPENAI_API_KEY, which this project never provisions (per its own "use classical
ML / open models, not a default OpenAI dependency" stance). The judge LLM is
wired to whichever HF Inference Providers model this project already uses for
real generation (agent/nodes/generate.py's HuggingFaceProvider), reached via its
OpenAI-compatible router endpoint so LangChain's ChatOpenAI class can drive it
without actually depending on OpenAI. Embeddings reuse the exact same local
all-MiniLM-L6-v2 model the retrieval pipeline itself runs on
(ingestion/vector_store.py) — no extra model, no extra cost.

Run: python -m eval.ragas_eval
Run against the deployed HF Space instead of local generation:
    python -m eval.ragas_eval --production [space_url]
"""
from __future__ import annotations

import os
import sys

from agent.nodes.generate import DEFAULT_HF_MODEL, HuggingFaceProvider, generate_node
from ingestion.vector_store import VectorStore
from tools.rag_tools import retrieve_pdf_chunks

DEFAULT_SPACE_URL = "https://bhavyabhandary-finbuddy-langgraph-agent.hf.space"

# Real questions against the real ingested corpus (data/raw_pdfs/), each paired
# with a ground_truth fact extracted directly from the source PDF text (verified
# by grep against the actual PDF, not recalled from memory) — see this module's
# git history for the extraction. RAGAS scores are only meaningful against real
# retrieval output, so this replaced the earlier stub once real ingestion (
# milestone step 2) made that possible.
REAL_QA_SEEDS: list[dict] = [
    {
        "question": "What is the cap on Default Loss Guarantee (DLG) cover on a loan portfolio?",
        "ground_truth": (
            "The total amount of DLG cover on any outstanding portfolio, specified "
            "upfront, shall not exceed five per cent of the total amount disbursed "
            "out of that loan portfolio at any given time."
        ),
    },
    {
        "question": "What must consent from a Data Principal look like under the DPDP Act?",
        "ground_truth": (
            "Consent must be free, specific, informed, unconditional and unambiguous "
            "with a clear affirmative action, and must signify agreement to processing "
            "personal data for a specified purpose, limited to the personal data "
            "necessary for that purpose."
        ),
    },
    {
        "question": "Where must payment system data be stored under RBI's data storage rules?",
        "ground_truth": (
            "All system providers must ensure that the entire data relating to "
            "payment systems operated by them, including full end-to-end "
            "transaction details, is stored in a system only in India (the foreign "
            "leg of a transaction, if any, may also be stored abroad)."
        ),
    },
]


def build_real_qa_pairs(vector_store: VectorStore | None = None) -> list[dict]:
    """Runs real retrieval + real generation for each seed question — no mocking,
    no canned answers. Returns RAGAS's expected {question, contexts, answer,
    ground_truth} shape.
    """
    vector_store = vector_store or VectorStore()
    provider = HuggingFaceProvider()
    qa_pairs = []
    for seed in REAL_QA_SEEDS:
        retrieval = retrieve_pdf_chunks(seed["question"], vector_store)
        state = {
            "query": seed["question"],
            "route": "policy",
            "retrieved_chunks": retrieval.chunks,
            "top_score": retrieval.top_score,
            "sufficient": retrieval.sufficient,
        }
        result = generate_node(state, provider=provider)
        qa_pairs.append(
            {
                "question": seed["question"],
                "contexts": [c.text for c in retrieval.chunks],
                "answer": result["answer"],
                "ground_truth": seed["ground_truth"],
            }
        )
    return qa_pairs


def build_real_qa_pairs_from_production(
    space_url: str = DEFAULT_SPACE_URL, vector_store: VectorStore | None = None
) -> list[dict]:
    """Same seed questions, but the answer comes from a real HTTP call to the
    deployed HF Space's /agent/run -- genuinely production, not a local stand-in.

    One honest limitation: AgentRunResponse (api/main.py) doesn't expose the raw
    retrieved chunk text, only source filenames -- RAGAS needs the actual context
    strings for faithfulness/context_precision/context_recall. `contexts` here is
    reconstructed via local retrieve_pdf_chunks() against the *same* chroma_data/
    that was just deployed (verified identical: this ran right after confirming
    the redeployed Space's push landed the rebuilt corpus) -- retrieval is
    deterministic given the same corpus + embedding model + query, so this is
    what production's own internal retrieval used, not an approximation of it.
    Only `answer` is a genuine round-trip through the deployed instance.
    """
    import requests

    vector_store = vector_store or VectorStore()
    qa_pairs = []
    for seed in REAL_QA_SEEDS:
        retrieval = retrieve_pdf_chunks(seed["question"], vector_store)
        response = requests.post(
            f"{space_url}/agent/run", json={"query": seed["question"]}, timeout=90.0
        )
        response.raise_for_status()
        answer = response.json()["answer"]
        qa_pairs.append(
            {
                "question": seed["question"],
                "contexts": [c.text for c in retrieval.chunks],
                "answer": answer,
                "ground_truth": seed["ground_truth"],
            }
        )
    return qa_pairs


def _default_ragas_llm():
    """HF Inference Providers via its OpenAI-compatible endpoint, reusing the
    same model/token this project's own HuggingFaceProvider uses for real
    generation — never OpenAI's API, which this project doesn't provision.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=DEFAULT_HF_MODEL,
        api_key=os.environ.get("HF_TOKEN"),
        base_url=f"https://router.huggingface.co/{os.environ.get('HF_PROVIDER', 'featherless-ai')}/v1",
    )
    return LangchainLLMWrapper(chat)


def _default_ragas_embeddings():
    """The exact local embedding model the retrieval pipeline itself runs on
    (ingestion/vector_store.py) — no OpenAI, no second model to download.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


def evaluate_rag_quality(qa_pairs: list[dict], llm=None, embeddings=None) -> dict:
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
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm or _default_ragas_llm(),
        embeddings=embeddings or _default_ragas_embeddings(),
    )
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    if "--production" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--production"]
        space_url = args[0] if args else DEFAULT_SPACE_URL
        print(f"Running against deployed Space: {space_url}\n")
        pairs = build_real_qa_pairs_from_production(space_url)
    else:
        pairs = build_real_qa_pairs()
    for p in pairs:
        print(f"Q: {p['question']}\nA: {p['answer']}\n")
    scores = evaluate_rag_quality(pairs)
    print(scores)
