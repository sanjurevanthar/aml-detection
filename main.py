"""Single entry point for the AML detection experiment.

    python main.py            # bundled sample, no download needed
    python main.py --full     # full HI-Small file (must be downloaded first)

The same code path runs both modes; only the input file and output directory
change. All settings live in config.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    LOGREG_PARAMS,
    MAX_DAY,
    RESULTS_FULL,
    RESULTS_SAMPLE,
    SAMPLE_DATA_PATH,
    FULL_DATA_PATH,
    TOP_FRACTION,
    TOP_K,
)
from src.data import clean, load_raw, quality_report
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from src.metrics import (
    confusion_at_threshold,
    ranking_metrics,
    threshold_from_val,
)
from src.rules import fit_thresholds, score as rule_score
from src.split import select_period, time_split


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _plot_comparison(metrics: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics)
    values = [metrics[name]["recall_top_1pct"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"][: len(names)]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Recall among the riskiest 1% of test transactions")
    ax.set_title("Test set: recall at top 1% (higher is better)")
    ax.set_ylim(0, max(0.05, max(values) * 1.25))
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(full: bool) -> None:
    data_path = FULL_DATA_PATH if full else SAMPLE_DATA_PATH
    out_dir = RESULTS_FULL if full else RESULTS_SAMPLE
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {'full' if full else 'sample'} | input: {data_path}")
    df = clean(load_raw(data_path))
    df = select_period(df, max_day=MAX_DAY)
    report = quality_report(df)
    (out_dir / "data_summary.json").write_text(json.dumps({
        k: str(v) if isinstance(v, (pd.Timestamp, np.integer, np.floating)) else v
        for k, v in report.items()
    }, indent=2))

    print("Building past-only features ...")
    featured = build_features(df)
    train, val, test = time_split(featured)
    print(
        f"Split rows: train={len(train):,}, val={len(val):,}, test={len(test):,} "
        f"(day ranges from config.SPLIT_DAYS)"
    )

    # Rule baseline: thresholds come from train only.
    thresholds = fit_thresholds(train)
    rules_val, _ = rule_score(val, thresholds)
    rules_test, _ = rule_score(test, thresholds)

    # First ML comparator: one-hot/categorical fitting happens inside the
    # Pipeline, so it sees training rows only.
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = _make_preprocessor()
    model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(**LOGREG_PARAMS)),
        ]
    )
    print("Fitting logistic regression ...")
    model.fit(train[features], train["is_laundering"])
    logreg_val = model.predict_proba(val[features])[:, 1]
    logreg_test = model.predict_proba(test[features])[:, 1]

    candidates = [
        ("rules", rules_val, rules_test),
        ("logistic_regression", logreg_val, logreg_test),
    ]
    metrics: dict[str, dict] = {}
    for name, scores_val, scores_test in candidates:
        cutoff = threshold_from_val(val["is_laundering"], scores_val, TOP_FRACTION)
        entry = ranking_metrics(
            test["is_laundering"].to_numpy(), scores_test, TOP_FRACTION, TOP_K
        )
        entry["threshold_from_val"] = cutoff
        entry["confusion_at_threshold"] = confusion_at_threshold(
            test["is_laundering"].to_numpy(), scores_test, cutoff
        )
        metrics[name] = entry

    comparison_rows = []
    for name, entry in metrics.items():
        row = {"method": name}
        row.update({k: v for k, v in entry.items() if k != "confusion_at_threshold"})
        row.update({f"cm_{k}": v for k, v in entry["confusion_at_threshold"].items()})
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    _plot_comparison(metrics, out_dir / "recall_at_top_1pct.png")

    # Coefficients give a first, global explanation of the linear model.
    transformed_names = model.named_steps["prep"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    coef_table = (
        pd.DataFrame({"feature": transformed_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda t: t["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(25)
    )
    coef_table.to_csv(out_dir / "logreg_coefficients.csv", index=False)

    # Save row-level scores only for the tiny sample, so the evidence stays
    # inspectable without committing a large prediction file.
    if not full:
        scores = test[["row_id", "timestamp", "is_laundering"]].copy()
        scores["rule_score"] = rules_test
        scores["logreg_score"] = logreg_test
        scores.to_csv(out_dir / "test_scores.csv", index=False)

    print(f"\nThresholds: amount>= {thresholds.large_amount:,.2f}, "
          f"velocity>= {thresholds.velocity:,.1f} txns/hour")
    for name, entry in metrics.items():
        print(
            f"{name:20s} PR-AUC={entry['pr_auc']:.4f}  "
            f"ROC-AUC={entry['roc_auc']:.4f}  "
            f"recall@1%={entry['recall_top_1pct']:.4f}  "
            f"precision@1%={entry['precision_top_1pct']:.4f}"
        )
    print(f"Wrote results to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="use data/raw/HI-Small_Trans.csv and write to results/full/",
    )
    args = parser.parse_args()
    run(full=args.full)


if __name__ == "__main__":
    main()
