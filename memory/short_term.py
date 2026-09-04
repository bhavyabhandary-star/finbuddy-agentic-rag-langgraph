"""Short-term memory: the LangGraph state carried across one session's loop.

This module is intentionally thin — short-term memory here IS the LangGraph
AgentState (see agent/state.py), cleared per session. This file exists as the
named home for that decision and for the summarization helper the cost/latency
section calls for once conversations get long enough to need it.
"""
from __future__ import annotations


def summarize_history_if_needed(messages: list[str], max_messages: int = 6) -> list[str]:
    """Per kickoff_prompt.md's token-minimization checklist: summarize conversational
    memory instead of replaying full history.

    TODO: replace the truncation below with a real LLM/extractive summary once
    conversations in practice exceed max_messages — truncating is a safe, cheap
    placeholder that never fabricates a summary, unlike a stub summary would.
    """
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
