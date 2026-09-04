"""Minimal, directory-based model registry for the Intent Router.

staging -> human sign-off -> production; old versions archived, never overwritten.
A versioned local directory is a legitimate MVP registry (build_prompt.md doesn't
require a hosted registry) as long as the state transitions are explicit and
auditable — that's what this module enforces.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_ROOT = Path(__file__).parent / "registry_store"
STAGES = ("staging", "production", "archived")


def _stage_dir(stage: str) -> Path:
    assert stage in STAGES, f"unknown stage {stage!r}"
    path = REGISTRY_ROOT / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def promote_to_staging(artifact_path: Path, metrics: dict) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest_dir = _stage_dir("staging") / version
    dest_dir.mkdir(parents=True)
    shutil.copy(artifact_path, dest_dir / artifact_path.name)
    with open(dest_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return dest_dir


def promote_to_production(staging_version_dir: Path, signed_off_by: str) -> Path:
    """Requires an explicit human sign-off name — never auto-promoted.

    Per build_prompt.md's governance gate: "New router version must beat the
    current one on the eval harness and get human sign-off before serving."
    """
    if not signed_off_by:
        raise ValueError("promote_to_production requires a non-empty signed_off_by")

    # Archive whatever is currently in production before replacing it.
    current_prod = _stage_dir("production")
    if any(current_prod.iterdir()):
        archive_dest = _stage_dir("archived") / current_prod.name
        shutil.move(str(current_prod), str(archive_dest))
        current_prod.mkdir(parents=True)

    version_name = staging_version_dir.name
    dest = current_prod / version_name
    shutil.copytree(staging_version_dir, dest)
    with open(dest / "signoff.json", "w") as f:
        json.dump(
            {"signed_off_by": signed_off_by, "signed_off_at": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )
    return dest


def get_current_production_artifact() -> Path | None:
    prod_dir = _stage_dir("production")
    versions = sorted(prod_dir.iterdir())
    if not versions:
        return None
    latest = versions[-1]
    matches = list(latest.glob("*.joblib"))
    return matches[0] if matches else None
