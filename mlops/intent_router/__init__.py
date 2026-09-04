"""The Intent Router — the one model this project fully owns and governs.

Decides, per incoming message, which of three paths the agent takes:
policy/coaching question (RAG) | credit-assessment request | off-topic.
Never confused with the five capstone models (build_prompt.md's correction note).
"""
