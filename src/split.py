"""Time-based train/validation/test split by day."""

from __future__ import annotations

import pandas as pd

from config import MAX_DAY, SPLIT_DAYS


def select_period(df: pd.DataFrame, max_day: int = MAX_DAY) -> pd.DataFrame:
    """Drop the anomalous high-laundering tail after ``max_day``."""
    return df.loc[df["day"] <= max_day].copy()


def time_split(
    df: pd.DataFrame,
    days: dict[str, tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bounds = days or SPLIT_DAYS
    masks = {}
    for name, (first, last) in bounds.items():
        masks[name] = df["day"].between(first, last)
    covered = masks["train"] | masks["val"] | masks["test"]
    if not bool(covered.all()):
        raise ValueError(
            f"Split days do not cover all rows: {int((~covered).sum())} rows "
            f"have day values outside {bounds}"
        )
    return (
        df.loc[masks["train"]].copy(),
        df.loc[masks["val"]].copy(),
        df.loc[masks["test"]].copy(),
    )
