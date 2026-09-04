"""MLOps Layer (build_prompt.md, Layer 7) — scoped to what this project owns.

intent_router/       — full train/track/registry/drift/retrain loop for the
                        Intent Router, the one model this project actually creates.
risk_trend_monitor/  — input-drift DETECTION ONLY for the read-only Risk-Trend
                        tool; retraining that model is production FinBuddy's job,
                        not this project's.

This project deliberately does NOT re-implement MLOps for F-001/F-003/F-006/F-012 —
that governance already exists in finbuddy-project and duplicating it here would be
redundant, not additive. See build_prompt.md's "MLOps — scoped to what this project
actually owns" section.
"""
