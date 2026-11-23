"""Build the small bundled sample from the full local HI-Small CSV.

Usage (full CSV must already be present): python -m src.make_sample
Selection is deterministic: every 100th row from days 0-9, in original row
order. This preserves both the time structure and the ~0.1% laundering rate.
"""

from config import FULL_DATA_PATH, MAX_DAY, SAMPLE_DATA_PATH, SAMPLE_MODULUS
from src.data import COLS, clean, load_raw


def main() -> None:
    df = clean(load_raw(FULL_DATA_PATH))
    df = df.loc[df["day"] <= MAX_DAY]
    sample = df.loc[df["row_id"] % SAMPLE_MODULUS == 0, COLS].copy()
    # Match the raw file's timestamp format so load_raw() works unchanged.
    sample["timestamp"] = sample["timestamp"].dt.strftime("%Y/%m/%d %H:%M")

    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Saved {len(sample):,} rows to {SAMPLE_DATA_PATH}")
    print(f"Laundering rows: {int(sample['is_laundering'].sum()):,}")
    print(f"Timestamp range: {sample['timestamp'].min()} to {sample['timestamp'].max()}")
    print(f"Size: {SAMPLE_DATA_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
