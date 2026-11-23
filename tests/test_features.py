"""Causality checks for past-only features (stdlib unittest, no pytest needed)."""

import unittest

import numpy as np
import pandas as pd

from src.features import build_features


def frame(timestamps, amounts, accounts=None, counterparties=None):
    n = len(timestamps)
    accounts = accounts or ["A"] * n
    counterparties = counterparties or ["C"] * n
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, format="%H:%M"),
            "row_id": np.arange(len(timestamps)),
            "src_bank": ["010"] * len(timestamps),
            "src_acct": accounts,
            "dst_bank": ["020"] * len(timestamps),
            "dst_acct": counterparties,
            "amt_paid": amounts,
            "amt_received": amounts,
            "pay_ccy": ["US Dollar"] * len(timestamps),
            "recv_ccy": ["US Dollar"] * len(timestamps),
            "pay_format": ["Wire"] * len(timestamps),
            "is_laundering": [0] * len(timestamps),
        }
    )


HISTORY_PREFIXES = ("src_", "dst_")


def history_columns(df):
    return [
        c for c in df.columns
        if c.startswith(HISTORY_PREFIXES)
        and any(t in c for t in ("past_", "count_", "minutes_since_last", "has_history"))
    ]


class FeatureCausalityTests(unittest.TestCase):
    def test_future_row_does_not_change_earlier_history(self):
        before = build_features(frame(["00:00", "00:01", "00:02"], [10, 20, 100]))
        after = build_features(frame(["00:00", "00:01", "00:02"], [10, 20, 999999]))
        cols = history_columns(before)
        self.assertGreater(len(cols), 10)
        for i in (0, 1):
            for col in cols:
                left, right = before.loc[i, col], after.loc[i, col]
                if pd.isna(left) or pd.isna(right):
                    self.assertTrue(pd.isna(left) and pd.isna(right), f"{col} changed at row {i}")
                else:
                    self.assertEqual(left, right, f"{col} changed at row {i}")

    def test_same_minute_activity_is_not_history(self):
        df = build_features(frame(["00:00", "00:00", "00:01"], [10, 20, 30]))
        # The second row must not count the first row from the same minute.
        self.assertEqual(df.loc[1, "dst_past_count"], 0.0)
        self.assertEqual(df.loc[1, "dst_past_distinct_cp"], 0.0)
        # The next minute sees both earlier rows.
        self.assertEqual(df.loc[2, "dst_past_count"], 2.0)

    def test_first_transaction_has_no_history(self):
        df = build_features(frame(["00:00"], [10]))
        self.assertEqual(df.loc[0, "dst_past_count"], 0.0)
        self.assertEqual(df.loc[0, "dst_has_history"], 0.0)


if __name__ == "__main__":
    unittest.main()
