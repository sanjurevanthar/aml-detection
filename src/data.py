# src/data.py
import pandas as pd

COLS = [
    "timestamp", "src_bank", "src_acct", "dst_bank", "dst_acct",
    "amt_received", "recv_ccy", "amt_paid", "pay_ccy",
    "pay_format", "is_laundering",
]

def load_raw(path):
    """Read the raw CSV with clean column names; keep original order in row_id."""
    df = pd.read_csv(path, header=0, names=COLS)
    df["row_id"] = range(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    return df

def quality_report(df):
    """Print and return basic data-quality numbers."""
    report = {
        "rows": len(df),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=[c for c in df.columns if c != "row_id"]).sum()),
        "first_timestamp": df["timestamp"].min(),
        "last_timestamp": df["timestamp"].max(),
        "label_values": sorted(df["is_laundering"].unique().tolist()),
        "zero_or_negative_paid": int((df["amt_paid"] <= 0).sum()),
        "self_transfers": int((df["src_acct"] == df["dst_acct"]).sum()),
        "laundering_rate": float(df["is_laundering"].mean()),
    }
    for k, v in report.items():
        print(f"{k}: {v}")
    return report

def clean(df):
    """Sort by time (ties keep original order) and add a day index."""
    df = df.sort_values(["timestamp", "row_id"], kind="stable").reset_index(drop=True)
    d = df["timestamp"].dt.normalize()
    df["day"] = (d - d.min()).dt.days
    return df
