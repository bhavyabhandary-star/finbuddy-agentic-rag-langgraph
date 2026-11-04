"""Real-data readiness for the four credit models (F-001, F-003, F-006, Risk-Trend).

Production FinBuddy's own data generator says it plainly: every AUC it reports "is a statement
about this generator", and the models "must be retrained on real, consented Setu Account
Aggregator data" before any real lending decision relies on them. This package is how this
project works toward that and -- just as important -- how it refuses to claim it early:

  profiles   load + validate real Setu AA profiles (and record where they came from)
  outcomes   join observed repayment outcomes / trend labels (none exist yet)
  shift      how far real profiles sit from the data the models were trained on
  readiness  the gate: which models MAY be retrained on what we hold, and exactly what is missing
  export     writes the joined training table the production trainers can consume -- only if the gate passes
  collect    CLI that pulls a fresh consented profile and records it in the consent register

Nothing here retrains or replaces a production model. It decides whether anyone may.
"""
