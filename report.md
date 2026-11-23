# Project 2 Report — Money Laundering Classification in Bank Transactions

**Status: interim draft (23 November 2026).** This file is the official final
report location and will be expanded to 2,500–3,500 words before 7 December.
Numbers below are labelled by dataset and configuration; no result is reported
before it has been produced by `main.py`.

## Abstract (draft)

Banks screen money laundering with fixed rules that generate many false alerts.
We ask whether a risk-ranked machine-learning model can find at least twice as
many laundering transactions as a rule baseline when investigators review only
the top 1% of alerts. We use the IBM HI-Small dataset (5.08 million
transactions, ~0.1% labelled laundering) and split it by time: train on days
0–5, validate on days 6–7, and test on days 8–9. Features are causal — each
transaction sees only account activity from strictly earlier minutes — and all
learned thresholds/preprocessing are fitted on training days only. The current
iteration compares a five-rule baseline with class-weighted logistic
regression. On the full test set (862,792 transactions, 956 laundering), the
model reaches PR-AUC 0.0175 and recall@1% 0.233 versus 0.0013 and 0.0094 for
the rules — about 25× more laundering caught in the top 1%, clearing the 2×
benchmark. Precision@1% is still low (2.6% vs 0.10%), so the ranked list is
useful but far from a clean alert queue. The bundled 203,095-row sample is
reported separately as a functionality check. A major limitation is the
anomalous tail of the raw file (days 10–17), which is excluded from the primary
experiment.

## Introduction

Criminals move money through many accounts, banks, and currencies to disguise
its origin. Fixed thresholds such as “flag transfers over $10,000” are easy to
evade by splitting payments and flood investigators with innocent alerts. The
practical task is ranking: given a fixed review budget, put the most suspicious
transactions at the top. Our precise question is whether the best model's
recall in the top 1% is at least 2× the rule baseline's on the same held-out
test days.

## Background / related work

Rule engines remain common in banking compliance; the baseline mirrors their
logic (large amount, round amount, currency change, cash/Bitcoin, high account
velocity). Supervised tabular models—logistic regression, random forests, and
gradient boosting such as LightGBM—are standard alternatives; the proposal also
lists an unsupervised Isolation Forest. Our contribution is the independent,
leakage-safe implementation and the like-for-like comparison on a time-based
holdout, not a new algorithm. Data source: IBM Research & ETH Zurich AML
simulator via Kaggle; the Databricks ingest prototype in this repository
documents the original schema.

## Data

The raw file has 5,078,345 rows and 11 columns: timestamp, sending/receiving
bank and account, paid/received amounts and currencies, payment format, and
`is_laundering`. `src/data.py` renames the collision-prone columns, parses
timestamps, and assigns a stable `row_id`; there are no missing values, nine
exact duplicate rows, and a ~0.102% laundering rate overall. EDA tables and
figures are committed under `results/full/`.

Days 10–17 contain only 1,108 transactions but 657 laundering labels (~59%),
unlike days 0–9 (~0.089%). We restrict the primary experiment to days 0–9, the
roughly 10-day period described in the proposal, and document this choice
rather than mixing two regimes. The bundled sample is every 25th row of that
period (203,095 rows, 179 positive), selected deterministically to preserve
time order and class balance. The full CSV is ~454 MB and is not committed;
`data/download.py` documents acquisition.

## Methodology

**Split.** Day indices, fixed in `config.py`: train 0–5, validation 6–7, test
8–9. Validation chooses the alert cutoff; the test set is used once per method.

**Features.** `src/features.py` builds transaction features (log amounts,
amount ratio, currency mismatch, round amounts, self/bank mismatch, payment
format flags, hour/day-of-week) and account-history features for sender and
receiver: past count/sum/mean/max log amount, minutes since previous activity,
counts in the last 1 and 24 hours, and distinct counterparties seen before.
All history lookups use `merge_asof` with `allow_exact_matches=False`, so a
transaction never sees its own minute. The label is never used in features.

**Baseline.** Five flags are summed; the large-amount cutoff is the training
99th percentile and the velocity cutoff the training 99.9th percentile of
1-hour activity. A tiny amount tie-break makes ranking deterministic.

**Comparator.** Class-weighted logistic regression on numeric features plus
one-hot encoded payment format/currencies. Imputation and encoding live inside
a scikit-learn `Pipeline`, so they are fitted on training rows only.

**Metrics.** PR-AUC (primary ranking quality), ROC-AUC, recall and precision
within the riskiest 1%, precision at the top 100 alerts, and a confusion matrix
at the validation-chosen cutoff. Accuracy is not used for a 0.1% positive
class. All methods are scored on identical test rows.

## Results

**Full experiment** (`python main.py --full`; 5,077,237 rows after excluding
days 10–17; 862,792 test rows with 956 laundering transactions):

| Method | PR-AUC ↑ | ROC-AUC ↑ | Recall@1% ↑ | Precision@1% ↑ | TP@val cutoff | FP@val cutoff |
|---|---:|---:|---:|---:|---:|---:|
| Rule baseline | 0.0013 | 0.6056 | 0.0094 | 0.0010 | 9 | 8,798 |
| Logistic regression | 0.0175 | 0.9466 | 0.2333 | 0.0258 | 412 | 22,222 |

Source: [comparison.csv](results/full/comparison.csv),
[metrics.json](results/full/metrics.json),
[figure](results/full/recall_at_top_1pct.png), and
[coefficients](results/full/logreg_coefficients.csv). Rules and the model were
scored on identical test rows; thresholds were fitted on training days and the
alert cutoff was chosen on validation days only.

The model found 223 of 956 test laundering transactions in the riskiest 1%
versus 9 for the rules — **24.8× more**, well beyond the 2× benchmark. Its
PR-AUC is ~13.6× the baseline and ROC-AUC rises from 0.61 to 0.95. The tradeoff
is alert volume: even the better model's top 1% contains ~22,000 innocent
transactions, so precision@1% is only 2.6%. The validation-chosen cutoff
flags ~2.6% of test rows and catches 43.1% of laundering at 1.8% precision.

**Sample run** (`python main.py`, 203,095 rows, 34,517 test rows, 38 positives):

| Method | PR-AUC ↑ | ROC-AUC ↑ | Recall@1% ↑ | Precision@1% ↑ |
|---|---:|---:|---:|---:|
| Rule baseline | 0.0012 | 0.5748 | 0.000 | 0.000 |
| Logistic regression | 0.0096 | 0.9074 | 0.079 | 0.009 |

Source: [comparison.csv](results/sample/comparison.csv),
[metrics.json](results/sample/metrics.json),
[figure](results/sample/recall_at_top_1pct.png), and
[row-level scores](results/sample/test_scores.csv).

The sample shows the same ranking pattern (ROC-AUC 0.91 vs 0.57), but its
0.079 recall@1% is only three caught transactions, so it is a functionality
check rather than evidence for the research question.

## Discussion

The full-data comparison answers the research question in the affirmative:
the logistic model catches ~25× more laundering in the top 1% than the rule
baseline. The gain is not a split artifact — both methods see the same 956
positive test rows, features are causal, and all learned statistics come from
training days. The rule engine's top-1% list is dominated by innocent
high-value or cash transactions; only 9 of 8,628 flagged rows are laundering.
The model is much better at ordering risk, but it is still not a clean alert
queue: 22,222 false positives accompany 412 true positives at the
validation-chosen cutoff, and precision@1% is 2.6%. Investigators would still
need to review many innocent accounts.

Two limitations qualify the result. First, the anomalous days 10–17 were
excluded; including their ~59% laundering rate would make the headline metric
hard to interpret and would not reflect the proposal's ~10-day scope. Second,
this iteration uses one linear model with no hyperparameter tuning, so the
final report will test whether tree-based models improve precision at fixed
recall and will examine which transaction types the model misses. The
deterministic causal-feature implementation was checked with unit tests that
change a future row and verify earlier history is unchanged.

## Conclusion

On the full HI-Small test period, a class-weighted logistic model using
past-only features ranked laundering far above the rule baseline: recall@1%
was 0.233 versus 0.009 (24.8×), and PR-AUC was 0.0175 versus 0.0013. The
2× research benchmark is therefore met, though the low precision@1% shows
that ranking improvements do not eliminate false-alert workload.

## References

1. IBM Research & ETH Zurich, *IBM Transactions for Anti-Money Laundering
   (AML) Dataset*, Kaggle, 2019. Dataset link in README.
2. IBM, *AML-Data GitHub Documentation and Simulator Overview*.
3. scikit-learn developers, *Pipeline and ColumnTransformer documentation*.
4. EAS 510, *Project 2 revised specification*, Fall 2026.

## Contributions and AI Assistance

**Tools used:** OpenCode with the Big Pickle model assisted with implementing
and debugging `src/features.py`, `src/rules.py`, `src/metrics.py`, `main.py`,
and documentation. We checked its output by running the sample and full
pipelines, inspecting saved metrics, and testing that changing a future row
does not change any earlier row's history features.

**Team contributions (to confirm with both members):** Sanju — rule-engine
branch and baseline design; Maithili — EDA branch and the feature/pipeline
branch described here. The final report will state individual contributions
precisely before submission. No experiments or citations in this draft are
fabricated.
