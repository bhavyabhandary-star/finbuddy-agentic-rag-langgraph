"""Seed labeled examples for the Intent Router.

TODO(milestone step 5): replace/expand with real or realistic labeled examples —
this starter set exists so train.py is runnable and testable immediately. Keep
this dataset versioned (per build_prompt.md's "Data & Feature" MLOps step) once
it grows past this seed.
"""
from __future__ import annotations

ROUTES = ("policy", "credit_assessment", "off_topic")

SEED_EXAMPLES: list[tuple[str, str]] = [
    # policy / coaching questions
    ("why do you need my UPI transaction data", "policy"),
    ("how long do you keep my data", "policy"),
    ("what is the RBI default loss guarantee cap", "policy"),
    ("why was my credit limit reduced", "policy"),
    ("how do I improve my repayment score", "policy"),
    ("what happens to my data if I close my account", "policy"),
    ("is my data shared with third parties", "policy"),
    ("what does DPDP consent mean for me", "policy"),
    # credit-assessment requests
    ("can you check my credit score", "credit_assessment"),
    ("what would my score be with this income", "credit_assessment"),
    ("am I approved for a loan", "credit_assessment"),
    ("score my UPI profile", "credit_assessment"),
    ("is my financial trend improving or getting worse", "credit_assessment"),
    ("what's my risk trend this month", "credit_assessment"),
    ("evaluate my creditworthiness", "credit_assessment"),
    ("how much can I borrow based on my transactions", "credit_assessment"),
    # off-topic
    ("what's the weather like today", "off_topic"),
    ("what stock should I invest in this week", "off_topic"),
    ("tell me a joke", "off_topic"),
    ("who won the cricket match", "off_topic"),
]


def load_training_examples() -> tuple[list[str], list[str]]:
    texts = [t for t, _ in SEED_EXAMPLES]
    labels = [label for _, label in SEED_EXAMPLES]
    return texts, labels
