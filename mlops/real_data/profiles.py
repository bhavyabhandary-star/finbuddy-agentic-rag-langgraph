"""Load and validate real Setu AA profiles."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from mlops.real_data.constants import ORIGINS_PRODUCTION, ORIGINS_SANDBOX, PLAUSIBLE_RANGE, SIGNALS


@dataclass
class LoadedProfiles:
    valid: list[dict] = field(default_factory=list)
    rejected: list[tuple[int, list[str]]] = field(default_factory=list)  # (line number, issues)
    duplicates: int = 0

    @property
    def origins(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.valid:
            counts[p.get("source", "unknown")] = counts.get(p.get("source", "unknown"), 0) + 1
        return counts

    @property
    def n_sandbox(self) -> int:
        return sum(1 for p in self.valid if p.get("source") in ORIGINS_SANDBOX)

    @property
    def n_production(self) -> int:
        return sum(1 for p in self.valid if p.get("source") in ORIGINS_PRODUCTION)


def validate_profile(profile: dict) -> list[str]:
    """Returns a list of problems; empty means the profile is usable."""
    issues: list[str] = []
    if not profile.get("user_id"):
        issues.append("missing user_id")
    source = profile.get("source")
    if source not in ORIGINS_SANDBOX | ORIGINS_PRODUCTION:
        issues.append(f"unknown source {source!r}")
    for name in SIGNALS:
        value = profile.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(f"{name} is missing or not numeric")
            continue
        low, high = PLAUSIBLE_RANGE[name]
        if not (low <= value <= high):
            issues.append(f"{name}={value} outside plausible range [{low}, {high}]")
    return issues


def load_real_profiles(path: str | Path | None = None) -> LoadedProfiles:
    target = Path(path) if path is not None else Path(settings.REAL_PROFILES_PATH)
    out = LoadedProfiles()
    if not target.exists():
        return out
    seen: set[str] = set()
    with open(target, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                profile = json.loads(line)
            except json.JSONDecodeError:
                out.rejected.append((lineno, ["not valid JSON"]))
                continue
            issues = validate_profile(profile)
            if issues:
                out.rejected.append((lineno, issues))
                continue
            key = str(profile["user_id"])
            if key in seen:
                out.duplicates += 1
                continue
            seen.add(key)
            out.valid.append(profile)
    return out
