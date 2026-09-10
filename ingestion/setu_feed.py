"""Setu AA Feed: consent-based UPI data pull, normalized to the same 8 signals
`assess_credit_profile` expects — the first arrow in the capstone's own
pipeline diagram: "Setu AA Feed -> FastAPI Inference -> SHAP + Calibration ->
Consumers." This module's job stops at producing those 8 signals; scoring
them is assess_credit_profile's job (it calls the live, already-governed API).

Two ways to get real signals here, both real, neither fabricated:

1. load_cached_real_profile() — reads data/setu_real_profiles.jsonl, a
   read-only copy of a genuinely real profile pulled through finbuddy-
   project's own verified Setu sandbox integration (real consent, real
   human approval via the mock-FIP webview, real decrypted transaction
   data, real normalization) — same "real external artifact, copied
   read-only" pattern as the RBI/DPDP PDFs and the Risk-Trend model.

2. SetuAAFeedClient — an INDEPENDENT implementation of Setu's real AA
   Gateway V2 consent+session protocol, written fresh in this project (not
   imported from finbuddy-project, per this project's separate-codebase
   principle) but informed by the real, hard-won protocol facts
   finbuddy-project's own client.py documents from a verified live call:
     - Auth is a two-step client-credentials exchange (POST
       https://orgservice-prod.setu.co/v1/users/login for a short-lived
       bearer JWT), not direct client-id/secret headers on every call.
     - The consent body needs an undocumented required field for Bridge's
       default "VIEW" consent mode: {"dataLife": {"unit": "MONTH",
       "value": "0"}} — omitting it 400s with "datalife value has to be 0".
     - `vua` must be "<mobile>@<AA-handle>" (e.g. "9999999999@onemoney"),
       NOT an FIP name — the FIP is chosen later, inside the consent
       webview.
     - The live data-session API returns per-account status under
       `FIstatus`, not `status` as Setu's own doc example shows.
   These are facts about calling a third-party API correctly, not
   finbuddy-project's proprietary code.

VERIFIED CONSTRAINT, not assumed: sandbox consent approval requires a human
to open `consent.url` and click through Setu's mock-FIP webview — there is no
headless/API way to auto-approve a sandbox consent (confirmed directly in
finbuddy-project's client.py docstring, itself based on a verified live run).
A fresh pull cannot be fully automated end-to-end by this project alone.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CACHED_PROFILE_PATH = os.environ.get("SETU_CACHED_PROFILE_PATH", "./data/setu_real_profiles.jsonl")

TOKEN_URL = "https://orgservice-prod.setu.co/v1/users/login"

# Same honest limitation finbuddy-project's normalizer documents: AA DEPOSIT
# data gives free-text bank narrations, not structured merchant/category
# tags. This is a heuristic, never ground truth.
_BUSINESS_KEYWORDS = re.compile(
    r"\b(?:gst|invoice|vendor|supplier|wholesale|b2b|trader|distributor|purchase\s*order)\b",
    re.IGNORECASE,
)
_LOW_BALANCE_THRESHOLD_INR = 500.0


def load_cached_real_profile(path: str = CACHED_PROFILE_PATH) -> dict | None:
    """Returns the most recently fetched real Setu sandbox profile's 8 UPI
    signals (plus metadata), or None if no profile has been pulled yet.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])  # most recent


class SetuAAConfigError(RuntimeError):
    pass


class SetuAAError(RuntimeError):
    def __init__(self, message: str, response: Any = None):
        super().__init__(message)
        self.response = response


@dataclass
class SetuAAConfig:
    base_url: str
    client_id: str
    client_secret: str
    product_instance_id: str

    @classmethod
    def from_env(cls) -> "SetuAAConfig":
        base_url = os.environ.get("SETU_BASE_URL", "").rstrip("/")
        client_id = os.environ.get("SETU_CLIENT_ID", "")
        client_secret = os.environ.get("SETU_CLIENT_SECRET", "")
        product_instance_id = os.environ.get("SETU_PRODUCT_INSTANCE_ID", "")
        missing = [
            name
            for name, val in [
                ("SETU_BASE_URL", base_url),
                ("SETU_CLIENT_ID", client_id),
                ("SETU_CLIENT_SECRET", client_secret),
                ("SETU_PRODUCT_INSTANCE_ID", product_instance_id),
            ]
            if not val
        ]
        if missing:
            raise SetuAAConfigError(
                "Missing Setu AA sandbox credentials in environment: " + ", ".join(missing)
            )
        return cls(base_url, client_id, client_secret, product_instance_id)


class SetuAAFeedClient:
    """Consent + FI-data-session flow against Setu's real AA Gateway V2 API.
    A fresh pull needs a human to approve consent.url — see module docstring.
    """

    def __init__(self, config: SetuAAConfig | None = None):
        self.config = config or SetuAAConfig.from_env()
        self._access_token: str | None = None

    def _fetch_access_token(self) -> str:
        import httpx

        resp = httpx.post(
            TOKEN_URL,
            headers={"Content-Type": "application/json", "client": "bridge"},
            json={"clientID": self.config.client_id, "grant_type": "client_credentials", "secret": self.config.client_secret},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise SetuAAError(f"Setu token exchange failed: {resp.status_code} {resp.text}", resp)
        token = resp.json().get("access_token")
        if not token:
            raise SetuAAError(f"Setu token exchange returned no access_token: {resp.text}", resp)
        return token

    def _headers(self) -> dict[str, str]:
        if self._access_token is None:
            self._access_token = self._fetch_access_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token}",
            "x-product-instance-id": self.config.product_instance_id,
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        import httpx

        url = f"{self.config.base_url}{path}"
        resp = httpx.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        if resp.status_code == 401 and self._access_token is not None:
            self._access_token = None
            resp = httpx.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise SetuAAError(f"Setu AA API {method} {path} failed: {resp.status_code} {resp.text}", resp)
        return resp.json() if resp.content else {}

    def create_consent(
        self, vua: str, duration_months: int = 12, data_range_from: str | None = None, data_range_to: str | None = None
    ) -> dict[str, Any]:
        """Returns {id, url, status, ...}. `url` is what a human must open to approve."""
        body: dict[str, Any] = {
            "consentDuration": {"unit": "MONTH", "value": str(duration_months)},
            "vua": vua,
            "context": [],
            "dataLife": {"unit": "MONTH", "value": "0"},  # undocumented but required — see module docstring
        }
        if data_range_from and data_range_to:
            body["dataRange"] = {"from": data_range_from, "to": data_range_to}
        return self._request("POST", "/v2/consents", json=body)

    def get_consent_status(self, consent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/consents/{consent_id}", params={"expanded": "true"})

    def wait_for_consent(self, consent_id: str, poll_seconds: int = 5, timeout_seconds: int = 600) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status = self.get_consent_status(consent_id)
            if status.get("status") in ("ACTIVE", "REJECTED", "REVOKED", "EXPIRED"):
                return status
            time.sleep(poll_seconds)
        raise TimeoutError(f"Consent {consent_id} did not resolve within {timeout_seconds}s")

    def create_data_session(self, consent_id: str, data_range_from: str, data_range_to: str) -> dict[str, Any]:
        body = {"consentId": consent_id, "dataRange": {"from": data_range_from, "to": data_range_to}, "format": "json"}
        return self._request("POST", "/v2/sessions", json=body)

    def get_data_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/sessions/{session_id}")

    def wait_for_data_session(self, session_id: str, poll_seconds: int = 5, timeout_seconds: int = 300) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            session = self.get_data_session(session_id)
            if session.get("status") == "COMPLETED":
                return session
            time.sleep(poll_seconds)
        raise TimeoutError(f"Data session {session_id} did not complete within {timeout_seconds}s")


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_deposit_account(account_data: dict, user_id: str) -> dict:
    """`account_data` is one session['fips'][i]['accounts'][j]['data']['account']
    entry from Setu's GET /v2/sessions/:id response, for an account of type
    'deposit'. See module docstring for the honest merchant_diversity/b2b_ratio
    limitations — carried forward here, not silently dropped.
    """
    import numpy as np
    import pandas as pd

    transactions = account_data.get("transactions", {}).get("transaction", [])
    if isinstance(transactions, dict):
        transactions = [transactions]
    if not transactions:
        raise ValueError(f"No transactions in AA payload for {user_id}; cannot normalize")

    rows = []
    for txn in transactions:
        try:
            rows.append(
                {
                    "timestamp": _parse_ts(txn["transactionTimestamp"]),
                    "amount": float(txn["amount"]),
                    "type": txn.get("type", "").upper(),
                    "narration": re.sub(r"\s+", " ", txn.get("narration", "") or "").strip().lower(),
                    "current_balance": float(txn["currentBalance"]) if txn.get("currentBalance") else None,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    df = pd.DataFrame(rows).sort_values("timestamp")
    if df.empty:
        raise ValueError(f"All transactions malformed for {user_id}; cannot normalize")
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    period_start, period_end = df["timestamp"].min(), df["timestamp"].max()
    tenure_months = min(max(1, round((period_end - period_start).days / 30.44)), 12)

    df["month"] = df["timestamp"].dt.to_period("M")
    credits = df[df["type"] == "CREDIT"]
    monthly = credits.groupby("month")["amount"].sum()
    avg_monthly_income = float(monthly.mean()) if not monthly.empty else 0.0
    if len(monthly) >= 2 and monthly.mean() > 0:
        income_regularity_score = float(np.clip(1 - monthly.std() / monthly.mean(), 0.0, 1.0))
    else:
        income_regularity_score = 0.0

    recent = df[df["timestamp"] >= period_end - pd.Timedelta(days=30)]
    tx_count_30d = int(len(recent))
    merchant_diversity = int(recent["narration"].nunique()) if not recent.empty else 0
    balance_dip_frequency = (
        int((df["current_balance"] < _LOW_BALANCE_THRESHOLD_INR).sum()) if df["current_balance"].notna().any() else 0
    )
    b2b_ratio = float(df["narration"].str.contains(_BUSINESS_KEYWORDS, na=False).mean()) if len(df) else 0.0
    avg_transaction_size = float(df["amount"].abs().mean())

    return {
        "user_id": user_id,
        "avg_monthly_income": round(avg_monthly_income, 2),
        "income_regularity_score": round(income_regularity_score, 4),
        "tx_count_30d": tx_count_30d,
        "merchant_diversity": merchant_diversity,
        "balance_dip_frequency": balance_dip_frequency,
        "b2b_ratio": round(b2b_ratio, 4),
        "avg_transaction_size": round(avg_transaction_size, 2),
        "tenure_months": tenure_months,
        "source": "setu_aa_sandbox",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_session_response(session: dict) -> list[dict]:
    """Walk every ready/delivered deposit account in a completed data session.
    Checks both `FIstatus` (the real live key) and `status` (what Setu's own
    doc example shows) — the doc example isn't reliable evidence of the live
    shape, per finbuddy-project's own verified finding.
    """
    profiles = []
    for fip in session.get("fips") or []:
        for account in fip.get("accounts", []):
            fi_status = account.get("FIstatus") or account.get("status")
            if fi_status not in ("READY", "DELIVERED"):
                continue
            acc_data = account.get("data", {}).get("account", {})
            if acc_data.get("type") != "deposit":
                continue
            link_ref = account.get("linkRefNumber", "unknown")
            try:
                profiles.append(normalize_deposit_account(acc_data, user_id=link_ref))
            except ValueError:
                continue
    return profiles


def pull_fresh_profile(vua: str, months: int = 12) -> dict:
    """End-to-end fresh pull. BLOCKS on human consent approval — prints the
    URL and polls; there is no way around that step (see module docstring).
    Appends the result to CACHED_PROFILE_PATH so load_cached_real_profile()
    picks it up afterward.
    """
    client = SetuAAFeedClient()
    now = datetime.now(timezone.utc)
    data_from = now - timedelta(days=30 * months)
    iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731

    consent = client.create_consent(vua=vua, duration_months=months, data_range_from=iso(data_from), data_range_to=iso(now))
    print(f"Open this URL and approve via the mock FIP webview:\n  {consent['url']}")
    print("Waiting for approval (polling every 5s, 10 min timeout)...")

    status = client.wait_for_consent(consent["id"])
    if status.get("status") != "ACTIVE":
        raise SetuAAError(f"Consent did not become ACTIVE (status={status.get('status')})")

    session = client.create_data_session(consent["id"], iso(data_from), iso(now))
    completed = client.wait_for_data_session(session["id"])
    profiles = normalize_session_response(completed)
    if not profiles:
        raise SetuAAError("No DELIVERED deposit accounts found in the session response")

    Path(CACHED_PROFILE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CACHED_PROFILE_PATH, "a", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile) + "\n")

    return profiles[0]
