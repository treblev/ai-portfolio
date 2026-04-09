"""
Synthetic ATO (Account Takeover) dataset generator for a Bitcoin exchange.

Each row is a withdrawal session event. The model answers:
  "Given this session's behavior, is this withdrawal being made by the
   legitimate account owner or an attacker?"

User profiles capture historical baselines. Features measure deviation
from those baselines — the core signal in ATO detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_USERS    = 10000
N_SESSIONS = 80000
FRAUD_RATE = 0.06       # ATO is rarer than general fraud; ~6% of withdrawal sessions
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"

KYC_TIERS   = ["unverified", "basic", "full"]
KYC_WEIGHTS = [0.20, 0.45, 0.35]

IP_COUNTRIES = ["US", "UK", "DE", "SG", "NG", "RU", "CN", "BR", "CA", "AU"]
# Attacker country distribution skews toward high-risk jurisdictions
ATTACKER_COUNTRY_WEIGHTS = [0.08, 0.06, 0.05, 0.07, 0.12, 0.18, 0.16, 0.08, 0.05, 0.05]
LEGIT_COUNTRY_WEIGHTS    = [0.25, 0.15, 0.12, 0.10, 0.05, 0.04, 0.04, 0.08, 0.10, 0.07]


# ── User profiles ──────────────────────────────────────────────────────────────

def build_users(n: int) -> pd.DataFrame:
    """
    Each user has a stable behavioral baseline derived from their history.
    These baselines are what ATO signals deviate from.
    """
    kyc = RNG.choice(KYC_TIERS, size=n, p=KYC_WEIGHTS)
    account_age_days = RNG.integers(30, 1825, size=n)

    # Typical withdrawal amount per user (BTC) — log-normal, KYC-scaled
    kyc_scale = np.where(kyc == "unverified", 0.05,
                np.where(kyc == "basic", 0.5, 3.0))
    typical_withdrawal = RNG.lognormal(mean=np.log(kyc_scale), sigma=0.6)

    # User's normal login hour (UTC) — each user has a personal pattern
    normal_hour_mean = RNG.uniform(6, 22, size=n)
    normal_hour_std  = RNG.uniform(1, 4, size=n)

    # Each user's usual IP country
    usual_country = np.array([
        RNG.choice(IP_COUNTRIES, p=LEGIT_COUNTRY_WEIGHTS) for _ in range(n)
    ])

    # Device pool per user (1–3 known devices)
    n_devices = RNG.integers(1, 4, size=n)

    # Average days between logins (activity level)
    avg_days_between_logins = RNG.exponential(scale=4.0, size=n)
    avg_days_between_logins = np.clip(avg_days_between_logins, 0.5, 60)

    return pd.DataFrame({
        "user_id":                  [f"U{i:07d}" for i in range(n)],
        "kyc_tier":                 kyc,
        "account_age_days":         account_age_days,
        "typical_withdrawal_btc":   np.round(typical_withdrawal, 6),
        "normal_hour_mean":         np.round(normal_hour_mean, 1),
        "normal_hour_std":          np.round(normal_hour_std, 1),
        "usual_country":            usual_country,
        "n_known_devices":          n_devices,
        "avg_days_between_logins":  np.round(avg_days_between_logins, 2),
    })


# ── Session generation ─────────────────────────────────────────────────────────

def _legit_session(user: pd.Series) -> dict:
    """Legitimate session: behavior consistent with user baseline."""
    hour = int(np.clip(
        RNG.normal(user["normal_hour_mean"], user["normal_hour_std"]), 0, 23
    ))
    amount = float(np.clip(
        RNG.lognormal(
            mean=np.log(max(user["typical_withdrawal_btc"], 1e-6)),
            sigma=0.4
        ),
        1e-5, user["typical_withdrawal_btc"] * 5
    ))
    days_since_login = float(RNG.exponential(user["avg_days_between_logins"]))
    session_to_withdrawal_secs = int(RNG.lognormal(mean=7.5, sigma=2.0))  # ~1800s median

    return {
        "is_new_device":               int(RNG.random() < 0.05),   # occasional new device
        "is_new_ip_country":           int(RNG.random() < 0.03),   # rare travel
        "days_since_last_login":       round(days_since_login, 2),
        "session_to_withdrawal_secs":  session_to_withdrawal_secs,
        "is_new_withdrawal_address":   int(RNG.random() < 0.10),
        "amount_btc":                  round(amount, 6),
        "hour_of_day":                 hour,
        "is_fraud":                    0,
    }


def _ato_session(user: pd.Series) -> dict:
    """
    ATO session: attacker deviates from the legitimate user's baseline.
    Multiple signals fire simultaneously — that co-occurrence is key.
    """
    # Attackers often operate at odd hours relative to victim's timezone
    hour = int(RNG.choice(
        [RNG.integers(0, 6), RNG.integers(22, 24)],   # early morning or late night
    ))

    # Large withdrawal — attacker wants maximum extraction
    amount = float(RNG.uniform(
        user["typical_withdrawal_btc"] * 2,
        user["typical_withdrawal_btc"] * 10
    ))

    # Account was dormant before attacker struck
    days_since_login = float(RNG.uniform(14, 120))

    # Attacker moves fast — straight to withdrawal, no browsing
    session_to_withdrawal_secs = int(RNG.lognormal(mean=6.6, sigma=1.5))  # ~55s median

    return {
        "is_new_device":               int(RNG.random() < 0.92),   # almost always new device
        "is_new_ip_country":           int(RNG.random() < 0.75),   # usually different country
        "days_since_last_login":       round(days_since_login, 2),
        "session_to_withdrawal_secs":  session_to_withdrawal_secs,
        "is_new_withdrawal_address":   int(RNG.random() < 0.95),   # always new destination
        "amount_btc":                  round(amount, 6),
        "hour_of_day":                 hour,
        "is_fraud":                    1,
    }


def build_sessions(users: pd.DataFrame, n: int, fraud_rate: float) -> pd.DataFrame:
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    rows = []

    for _ in range(n_legit):
        user = users.sample(1).iloc[0]
        session = _legit_session(user)
        session["user_id"]          = user["user_id"]
        session["kyc_tier"]         = user["kyc_tier"]
        session["account_age_days"] = user["account_age_days"]
        session["usual_country"]    = user["usual_country"]
        rows.append(session)

    for _ in range(n_fraud):
        user = users.sample(1).iloc[0]
        session = _ato_session(user)
        session["user_id"]          = user["user_id"]
        session["kyc_tier"]         = user["kyc_tier"]
        session["account_age_days"] = user["account_age_days"]
        session["usual_country"]    = user["usual_country"]
        rows.append(session)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df["session_id"] = [f"S{i:08d}" for i in range(len(df))]

    base_ts = pd.Timestamp("2025-01-01")
    offsets = pd.to_timedelta(RNG.integers(0, 90 * 24 * 60, size=len(df)), unit="min")
    df["timestamp"] = base_ts + offsets
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── Derived features ───────────────────────────────────────────────────────────

def add_derived_features(sessions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    df = sessions.merge(
        users[["user_id", "typical_withdrawal_btc", "normal_hour_mean", "normal_hour_std"]],
        on="user_id",
        how="left",
    )

    # How large is this withdrawal relative to the user's historical average
    df["amount_vs_user_avg"] = np.round(
        df["amount_btc"] / df["typical_withdrawal_btc"].clip(lower=1e-6), 4
    )

    # How far is the login hour from the user's normal window
    df["hour_deviation"] = np.round(
        np.abs(df["hour_of_day"] - df["normal_hour_mean"]), 2
    )

    # Encode KYC tier as ordinal
    kyc_map = {"unverified": 0, "basic": 1, "full": 2}
    df["kyc_tier_enc"] = df["kyc_tier"].map(kyc_map)

    df.drop(columns=["typical_withdrawal_btc", "normal_hour_mean", "normal_hour_std"],
            inplace=True)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building user profiles...")
    users = build_users(N_USERS)

    print(f"Generating {N_SESSIONS:,} sessions ({FRAUD_RATE*100:.0f}% ATO fraud)...")
    sessions = build_sessions(users, N_SESSIONS, FRAUD_RATE)

    print("Computing derived features...")
    sessions = add_derived_features(sessions, users)

    export_cols = [
        "session_id", "timestamp", "user_id",
        "kyc_tier", "kyc_tier_enc",
        "account_age_days",
        "is_new_device",
        "is_new_ip_country",
        "usual_country",
        "days_since_last_login",
        "session_to_withdrawal_secs",
        "is_new_withdrawal_address",
        "amount_btc",
        "amount_vs_user_avg",
        "hour_of_day",
        "hour_deviation",
        "is_fraud",
    ]

    out_path = OUTPUT_DIR / "ato_sessions.csv"
    sessions[export_cols].to_csv(out_path, index=False)
    print(f"Saved {len(sessions):,} rows → {out_path}")

    print("\nClass balance:")
    print(sessions["is_fraud"].value_counts(normalize=True)
          .rename({0: "legit", 1: "ato"}).to_string())

    print("\nATO vs legit — median feature comparison:")
    compare_cols = [
        "is_new_device", "is_new_ip_country", "days_since_last_login",
        "session_to_withdrawal_secs", "is_new_withdrawal_address",
        "amount_vs_user_avg", "hour_deviation",
    ]
    print(sessions.groupby("is_fraud")[compare_cols].median().T.rename(
        columns={0: "legit", 1: "ato"}
    ).to_string())


if __name__ == "__main__":
    main()
