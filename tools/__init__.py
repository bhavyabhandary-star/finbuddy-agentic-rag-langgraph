"""Tool Layer (build_prompt.md, Layer 3).

Four tools the agent can call:
  - retrieve_pdf_chunks / check_sufficiency / fetch_source_metadata  (RAG)
  - assess_credit_profile   -> calls production FinBuddy's live scoring API
                               (F-006 + F-001 + F-003 + F-012, read-only consumption)
  - assess_risk_trend       -> loads the registered Risk-Trend artifact read-only
"""
