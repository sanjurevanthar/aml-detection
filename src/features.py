"""Past-only transaction and account-history features.

Design rule: a feature for transaction t may only use rows whose timestamp is
*strictly earlier* than t's minute. The label ``is_laundering`` is never used
here, so there is no target leakage either. Waiting until the next minute keeps
activity inside the same minute out of the history, which avoids relying on the
arbitrary file order of ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES = [
    "log_amt_paid",
    "log_amt_received",
    "amt_ratio",
    "currency_mismatch",
    "round_100",
    "round_1000",
    "self_transfer",
    "bank_mismatch",
    "is_cash",
    "is_bitcoin",
    "hour",
    "dayofweek",
]

HISTORY_FEATURES = [
    "past_count",
    "past_sum_log_amt",
    "past_mean_log_amt",
    "past_max_log_amt",
    "minutes_since_last",
    "has_history",
    "count_1h",
    "count_24h",
    "past_distinct_cp",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    f"{prefix}_{name}" for prefix in ("src", "dst") for name in HISTORY_FEATURES
]

CATEGORICAL_FEATURES = ["pay_format", "pay_ccy", "recv_ccy"]

MAX_GAP_MINUTES = 7 * 24 * 60  # sentinel for an account's first observed transaction


def _past_key_features(
    df: pd.DataFrame,
    key: str,
    amount: str,
    counterparty: str,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Build past-only features for one account key (sender or receiver).

    ``df`` must be sorted by timestamp then row_id and have a RangeIndex.
    """
    work = df[[key, "timestamp", amount, counterparty]].copy()

    # Collapse to one row per (account, minute) first. Looking up the cumulative
    # total at the last minute *before* the current one makes the features strict.
    minute = (
        work.groupby([key, "timestamp"], sort=False)
        .agg(n=(amount, "size"), s=(amount, "sum"), mx=(amount, "max"))
        .reset_index()
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    grouped = minute.groupby(key, sort=False)
    minute["cum_n"] = grouped["n"].cumsum()
    minute["cum_s"] = grouped["s"].cumsum()
    minute["cum_mx"] = grouped["mx"].cummax()
    minute["prev_ts"] = grouped["timestamp"].shift(1)
    minute_right = minute[
        [key, "timestamp", "cum_n", "cum_s", "cum_mx", "prev_ts"]
    ].sort_values("timestamp", kind="mergesort")

    left = pd.DataFrame(
        {
            key: df[key].to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
            "_order": np.arange(len(df), dtype=np.int64),
        }
    ).sort_values("timestamp", kind="mergesort")

    merged = pd.merge_asof(
        left,
        minute_right,
        on="timestamp",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_order").reset_index(drop=True)

    past_n = merged["cum_n"].fillna(0).to_numpy(dtype=np.float64)
    past_s = merged["cum_s"].fillna(0).to_numpy(dtype=np.float64)
    past_mean = np.divide(
        past_s,
        past_n,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=past_n > 0,
    )
    gap = (
        df["timestamp"].reset_index(drop=True) - merged["prev_ts"].reset_index(drop=True)
    ).dt.total_seconds() / 60.0

    out: dict[str, np.ndarray] = {
        f"{prefix}_past_count": past_n,
        f"{prefix}_past_sum_log_amt": past_s,
        f"{prefix}_past_mean_log_amt": past_mean,
        f"{prefix}_past_max_log_amt": merged["cum_mx"].to_numpy(dtype=np.float64),
        f"{prefix}_minutes_since_last": gap.fillna(MAX_GAP_MINUTES).to_numpy(),
        f"{prefix}_has_history": (past_n > 0).astype(np.float64),
    }

    # Count in a trailing window = cumulative-before-now minus cumulative-at-cutoff.
    cum_right = minute[[key, "timestamp", "cum_n"]].sort_values(
        "timestamp", kind="mergesort"
    )
    for hours, label in ((1, "1h"), (24, "24h")):
        cutoff_left = left.copy()
        cutoff_left["timestamp"] = cutoff_left["timestamp"] - pd.Timedelta(hours=hours)
        cutoff = pd.merge_asof(
            cutoff_left,
            cum_right,
            on="timestamp",
            by=key,
            direction="backward",
            allow_exact_matches=True,
        )
        cum_at_cutoff = (
            cutoff.sort_values("_order")["cum_n"].fillna(0).to_numpy(dtype=np.float64)
        )
        out[f"{prefix}_count_{label}"] = np.maximum(past_n - cum_at_cutoff, 0.0)

    # Number of distinct counterparties first seen before the current minute.
    pairs = work[[key, counterparty, "timestamp"]].drop_duplicates(
        [key, counterparty], keep="first"
    )
    pairs = pairs.sort_values("timestamp", kind="mergesort")
    pairs["rank"] = (
        pairs.groupby(key, sort=False).cumcount().to_numpy(dtype=np.float64) + 1.0
    )
    pairs = (
        pairs[[key, "timestamp", "rank"]]
        .rename(columns={"timestamp": "first_ts"})
        .sort_values("first_ts", kind="mergesort")
    )
    distinct = pd.merge_asof(
        left,
        pairs,
        left_on="timestamp",
        right_on="first_ts",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    out[f"{prefix}_past_distinct_cp"] = (
        distinct.sort_values("_order")["rank"].fillna(0).to_numpy(dtype=np.float64)
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with causal transaction and account-history features added."""
    required = {
        "timestamp", "row_id", "amt_paid", "amt_received", "pay_ccy", "recv_ccy",
        "pay_format", "src_bank", "src_acct", "dst_bank", "dst_acct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features missing columns: {sorted(missing)}")

    work = df.reset_index(drop=True).copy()
    if not work["timestamp"].is_monotonic_increasing:
        work = work.sort_values(["timestamp", "row_id"], kind="mergesort").reset_index(drop=True)

    work["src_key"] = work["src_bank"].astype(str) + "_" + work["src_acct"].astype(str)
    work["dst_key"] = work["dst_bank"].astype(str) + "_" + work["dst_acct"].astype(str)
    work["log_amt_paid"] = np.log1p(work["amt_paid"].clip(lower=0))
    work["log_amt_received"] = np.log1p(work["amt_received"].clip(lower=0))

    history: dict[str, np.ndarray] = {}
    history.update(_past_key_features(work, "src_key", "log_amt_paid", "dst_key", "src"))
    history.update(_past_key_features(work, "dst_key", "log_amt_received", "src_key", "dst"))
    for name, values in history.items():
        work[name] = values

    work["amt_ratio"] = (work["amt_received"] / (work["amt_paid"] + 1.0)).clip(upper=100.0)
    work["currency_mismatch"] = (work["pay_ccy"] != work["recv_ccy"]).astype(float)
    work["round_100"] = ((work["amt_paid"] % 100 == 0) & (work["amt_paid"] > 0)).astype(float)
    work["round_1000"] = ((work["amt_paid"] % 1000 == 0) & (work["amt_paid"] > 0)).astype(float)
    work["self_transfer"] = (work["src_key"] == work["dst_key"]).astype(float)
    work["bank_mismatch"] = (work["src_bank"] != work["dst_bank"]).astype(float)
    work["is_cash"] = (work["pay_format"] == "Cash").astype(float)
    work["is_bitcoin"] = (work["pay_format"] == "Bitcoin").astype(float)
    work["hour"] = work["timestamp"].dt.hour.astype(float)
    work["dayofweek"] = work["timestamp"].dt.dayofweek.astype(float)
    return work
