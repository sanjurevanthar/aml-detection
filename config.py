"""Central configuration for the AML detection experiment.

Every number that affects an experiment lives here so a run can be described by
pointing at this file. Keep this module free of heavy imports: both ``main.py``
and the sample-generation script import it.
"""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Data -----------------------------------------------------------------------
# The IBM HI-Small file has an anomalous tail: days 10-17 contain only ~1.1k
# transactions but are ~59% laundering, unlike the ~0.1% rate on days 0-9.
# The primary investigation covers the ~10-day period promised in the proposal.
MAX_DAY = 9
FULL_DATA_PATH = Path("data/raw/HI-Small_Trans.csv")
SAMPLE_DATA_PATH = Path("data/sample/sample.csv")

# Deterministic sample: keep every Nth row (by original row order) from each
# retained day. This preserves the time structure and the class balance.
SAMPLE_MODULUS = 25

# Time-based split, by day index (inclusive bounds). Train < val < test.
SPLIT_DAYS = {
    "train": (0, 5),
    "val": (6, 7),
    "test": (8, 9),
}

# Output locations
RESULTS_SAMPLE = Path("results/sample")
RESULTS_FULL = Path("results/full")

# Evaluation -----------------------------------------------------------------
TOP_FRACTION = 0.01  # investigators review the riskiest 1% of transactions
TOP_K = 100          # secondary "precision at top 100 alerts" metric

# Rule baseline --------------------------------------------------------------
# Thresholds are learned from the training days only (see src/rules.py).
RULES_LARGE_AMOUNT_QUANTILE = 0.99
RULES_VELOCITY_QUANTILE = 0.999

# First ML comparator. Tuning/other models are added in later iterations.
LOGREG_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}
