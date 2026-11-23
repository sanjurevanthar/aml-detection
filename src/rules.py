"""Simple rule-based baseline, in the spirit of current bank screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RULES_LARGE_AMOUNT_QUANTILE, RULES_VELOCITY_QUANTILE


@dataclass(frozen=True)
class RuleThresholds:
    large_amount: float
    velocity: float


def fit_thresholds(train: pd.DataFrame) -> RuleThresholds:
    """Learn thresholds from training days only; never from val or test."""
    large = float(train["amt_paid"].quantile(RULES_LARGE_AMOUNT_QUANTILE))
    velocity = float(
        pd.concat(
            [train["src_count_1h"], train["dst_count_1h"]], ignore_index=True
        ).quantile(RULES_VELOCITY_QUANTILE)
    )
    return RuleThresholds(large_amount=large, velocity=velocity)


def score(df: pd.DataFrame, thresholds: RuleThresholds) -> tuple[np.ndarray, pd.DataFrame]:
    """Return a ranking score and the individual 0/1 rule flags."""
    flags = pd.DataFrame(index=df.index)
    flags["large_amount"] = (df["amt_paid"] >= thresholds.large_amount).astype(float)
    flags["round_amount"] = df["round_1000"].to_numpy(dtype=float)
    flags["currency_change"] = df["currency_mismatch"].to_numpy(dtype=float)
    flags["cash_or_bitcoin"] = (
        (df["is_cash"].to_numpy() + df["is_bitcoin"].to_numpy()) > 0
    ).astype(float)
    flags["high_velocity"] = (
        (df["src_count_1h"] >= thresholds.velocity)
        | (df["dst_count_1h"] >= thresholds.velocity)
    ).astype(float)

    # Tiny tie-break by amount so investigators get a deterministic ordering
    # among transactions with the same number of triggered rules.
    total = flags.sum(axis=1).to_numpy(dtype=float)
    score_array = total + 1e-6 * df["log_amt_paid"].to_numpy(dtype=float)
    return score_array, flags
