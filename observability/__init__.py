"""Observability Layer (Layer 6): tracing, evaluation hooks, cost/step logging.

Per kickoff_prompt.md: every node logs what it decided and why, keyed by a run id,
so a bad output traces back to the exact decision that caused it — and trace tool
selection specifically (the stated reason), not just the final tool chosen.
"""
