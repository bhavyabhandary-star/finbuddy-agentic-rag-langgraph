"""Train the Intent Router and promote the winning config to production, for
CI's eval gate to actually exercise a real model instead of silently falling
back to infer.py's crude keyword heuristic.

Why this exists: mlops/intent_router/registry_store/ is gitignored (a real,
already-signed-off model lives there on a developer's machine and gets
bundled into the HF Space deploy snapshot separately -- see
scripts/prepare_hf_space_deploy.sh), so it never exists in a fresh CI
checkout. Without it, get_current_production_artifact() returns None,
classify_intent() falls through to _keyword_fallback() (always
confidence=0.5), and the eval gate ends up scoring that fallback instead of
the real model -- verified directly: removing registry_store/ locally and
re-running the gate reproduces CI's exact output byte-for-byte.

This is deliberately NOT the same as a real production promotion:
promote_to_production() normally requires a human's sign-off name per
registry.py's governance gate ("must beat the current one on the eval
harness and get human sign-off before serving"). Run only inside a CI
job's fresh, disposable checkout, this writes to that job's own throwaway
filesystem and is discarded when the runner is torn down -- it never
touches, overwrites, or gets committed as the real signed-off model a
developer promotes locally. The "signed_off_by" string below says exactly
that, so it's never mistaken for a real human sign-off if anyone inspects
this run's registry_store.

Run: python -m mlops.intent_router.train_and_promote
"""
from __future__ import annotations

import sys

from mlops.intent_router.registry import promote_to_production, promote_to_staging
from mlops.intent_router.train import ARTIFACTS_DIR, select_and_save_best, train_and_log_all_configs

CI_SIGNOFF_LABEL = "ci: automated train+eval-gate run (ephemeral runner only, not a real sign-off)"


def main() -> int:
    results = train_and_log_all_configs()
    winner = select_and_save_best(results)
    metrics = results[winner]["metrics"]
    print(f"Trained and selected: {winner} -> {metrics}")

    artifact_path = ARTIFACTS_DIR / "intent_router.joblib"
    staging_dir = promote_to_staging(artifact_path, metrics)
    promote_to_production(staging_dir, signed_off_by=CI_SIGNOFF_LABEL)
    print(f"Promoted {winner} to this run's ephemeral production registry for the eval gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
