"""Download files from the IBM AML transactions dataset (Kaggle) into data/raw/.

Only the requested files are downloaded. The full dataset is ~20+ GB
(Medium/Large variants), so we never fetch all of it.

Usage (from the project root):
    python data/download.py                    # HI-Small_Trans.csv
    python data/download.py HI-Small_Patterns.txt
"""

import argparse
import shutil
from pathlib import Path

import kagglehub

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=["HI-Small_Trans.csv"])
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        target = DEST / name
        if target.exists():
            print(f"skip (exists): {target}")
            continue
        # Downloads just this file to the kagglehub cache, returns its path.
        cached = Path(kagglehub.dataset_download(DATASET, path=name))
        # Move (not copy) so the file isn't stored twice on disk.
        shutil.move(str(cached), target)
        print(f"saved: {target}")


if __name__ == "__main__":
    main()
