"""Is this machine ready to run the live Setu Account Aggregator pipeline? Prints names and verdicts, never values.

Run: python -m mlops.real_data.preflight
"""
from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv

REQUIRED = ("SETU_BASE_URL", "SETU_CLIENT_ID", "SETU_CLIENT_SECRET", "SETU_PRODUCT_INSTANCE_ID")
OPTIONAL = ("SETU_WEBHOOK_SECRET", "HF_TOKEN", "LLM_PROVIDER")


def check(env=None) -> dict:
    env = os.environ if env is None else env
    missing = [name for name in REQUIRED if not env.get(name)]
    base = env.get("SETU_BASE_URL", "")
    host = urlparse(base).netloc
    return {
        "missing": missing,
        "ready": not missing,
        "base_url_host": host or None,
        "looks_like_sandbox": "sandbox" in host,
        "optional_set": [name for name in OPTIONAL if env.get(name)],
    }


def main() -> int:
    load_dotenv(find_dotenv(usecwd=True))
    r = check()
    print("SETU LIVE-PIPELINE PREFLIGHT")
    for name in REQUIRED:
        print(f"  {'ok     ' if name not in r['missing'] else 'MISSING'}  {name}")
    for name in OPTIONAL:
        print(f"  {'set    ' if name in r['optional_set'] else 'not set'}  {name} (optional)")
    if r["base_url_host"]:
        kind = "SANDBOX (mock bank accounts)" if r["looks_like_sandbox"] else "NOT a sandbox host: real borrower data, real consent obligations"
        print(f"\nSETU_BASE_URL host: {r['base_url_host']}  ->  {kind}")
    print("\n" + ("Ready." if r["ready"] else "Not ready: add the missing values to .env (never commit that file)."))
    return 0 if r["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
