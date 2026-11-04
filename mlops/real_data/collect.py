"""Pull one fresh, consented Setu AA profile and record the consent.

Needs SETU_* credentials in .env and a HUMAN to approve the consent in Setu's webview (there is no headless
way). The profile is appended to data/setu_real_profiles.jsonl and a consent-register line is written to
data/real_consent_register.jsonl. Both files are gitignored.

DPDP: collect only with the person's explicit consent for the stated purpose; keep the mobile / VUA as a hash,
never in clear; delete the profile on request or when the consent window ends. The register records what was
collected, why, when, and until when: the audit trail a Consent Manager would expect.

Run: python -m mlops.real_data.collect --vua 9999999999 [--months 12]
(--vua is the mobile alone, or <mobile>@<AA handle>. Setu's sandbox stopped accepting @onemoney in 2026-11; the bare mobile and @finvu work.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings
from mlops.real_data.readiness import assess, format_report


def register_consent(vua: str, profile: dict, months: int, purpose: str, path: Path | None = None) -> dict:
    now = datetime.now(timezone.utc)
    entry = {
        "user_id": profile["user_id"],
        "vua_sha256": hashlib.sha256(vua.encode("utf-8")).hexdigest(),
        "source": profile.get("source"),
        "purpose": purpose,
        "collected_at": now.isoformat(),
        "retain_until": (now + timedelta(days=30 * months)).isoformat(),
    }
    target = Path(path) if path is not None else Path(settings.REAL_CONSENT_REGISTER_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main() -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # the SETU_* credentials live in .env; nothing else loads it for a CLI run
    parser = argparse.ArgumentParser()
    parser.add_argument("--vua", required=True, help="mobile alone (e.g. 9999999999) or <mobile>@<AA handle> (e.g. 9999999999@finvu)")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--consent-id", default=None, help="resume a consent that is already approved instead of creating a new one")
    parser.add_argument("--session-id", default=None, help="with --consent-id: resume a data session that was already started")
    parser.add_argument("--purpose", default="Credit-model validation for the IIT Kanpur AIPL capstone (FinBuddy)")
    args = parser.parse_args()

    from ingestion.setu_feed import pull_fresh_profile  # imported late: it needs SETU_* env

    profile = pull_fresh_profile(args.vua, months=args.months, consent_id=args.consent_id, session_id=args.session_id)
    register_consent(args.vua, profile, args.months, args.purpose)
    print(f"Collected profile {profile['user_id']} (source={profile.get('source')}). Consent registered.\n")
    print(format_report(assess()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
