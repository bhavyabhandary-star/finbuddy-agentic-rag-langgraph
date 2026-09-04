"""Smoke tests: the graph compiles and off-topic queries never reach the LLM.

Full end-to-end tests (real retrieval, real Claude calls) belong in the eval
harness (eval/evaluate_agent.py), not here — these are fast, no-network unit tests.
"""
from unittest.mock import patch

from agent.graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_off_topic_query_escalates_without_llm_call():
    graph = build_graph()
    with patch("agent.nodes.route.classify_intent", return_value=("off_topic", 0.9)):
        with patch("agent.graph.generate_node") as mock_generate:
            result = graph.invoke({"query": "what's the weather today", "session_id": "test-session"})

    mock_generate.assert_not_called()
    assert result["escalate_to_human"] is True
