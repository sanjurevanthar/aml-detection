# Money Laundering Classification in Bank Transactions Using Machine Learning

EAS 510 Project 2 (Fall 2026) — team: **Maithili Yogesh Joshi** (50754842) and
**Sanju Authimahalingam Rajasri** (50742711).

**Research question.** If investigators can only review the riskiest 1% of
transactions, can a machine-learning model catch at least 2× more laundering
than a simple rule-based system?

The official report is [report.md](report.md). This README explains how to run
the code; the report explains what we found and what the evidence supports.

## Current status (interim, 23 Nov 2026)

- [x] Full IBM HI-Small CSV loaded locally and cleaned (`src/data.py`).
- [x] EDA and data-quality tables/figures saved under `results/full/`.
- [x] Causal, past-only transaction and account-history features (`src/features.py`).
- [x] Rule baseline with train-only thresholds (`src/rules.py`).
- [x] First ML comparator (logistic regression) and shared ranking metrics.
- [x] Single entry point `main.py` with a bundled, runnable sample.
- [x] Full-data run and preliminary baseline-vs-model comparison.
- [ ] LightGBM / random forest, tuning, SHAP, error analysis, final report.

## Environment

Tested with **Python 3.13.9 on macOS**. The sample run uses only CPU and finishes
in about **15 seconds** on the development machine; it needs no network access,
credentials, GPU, or paid service once dependencies are installed.

Pinned dependency versions are in [requirements.txt](requirements.txt):
`numpy==2.3.5`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.6`,
`joblib==1.5.2`, `kagglehub==1.0.3`.

## Exact commands

From a fresh clone, at the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

That is the default **sample** run. It writes to `results/sample/` and prints
PR-AUC, ROC-AUC, recall at the top 1%, and precision at the top 1% for both the
rule baseline and logistic regression. See
[results/sample/comparison.csv](results/sample/comparison.csv),
[results/sample/metrics.json](results/sample/metrics.json), and
[results/sample/recall_at_top_1pct.png](results/sample/recall_at_top_1pct.png).
Row-level scores are in
[results/sample/test_scores.csv](results/sample/test_scores.csv).

### Full experiment

The full CSV is ~454 MB and is intentionally **not** committed. Obtain it from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
then either use `kagglehub` (requires Kaggle credentials):

```bash
python data/download.py HI-Small_Trans.csv
```

or place the file at `data/raw/HI-Small_Trans.csv` yourself. Then run:

```bash
python main.py --full
```

This writes the report evidence to `results/full/` using the same code path.
It takes about **22 minutes** on the development machine (CPU, ~3 GB peak RSS).
Preliminary full-data outputs:
[comparison.csv](results/full/comparison.csv),
[metrics.json](results/full/metrics.json), and
[recall plot](results/full/recall_at_top_1pct.png).

To rebuild the bundled sample from the full file:

```bash
python -m src.make_sample
```

Optional sanity checks (standard library only):

```bash
python -m unittest discover -s tests -v
```

## Data and sample

- **Source:** IBM Transactions for Anti-Money Laundering, `HI-Small_Trans.csv`
  (IBM Research & ETH Zurich, via Kaggle). Each row is one bank transaction;
  `is_laundering` is the supervised target.
- **Full experimental window:** days 0–9 (2022-09-01 to 2022-09-10). The raw
  file also has days 10–17: only ~1,100 transactions, but ~59% laundering. That
  tail is unlike the ~0.1% rate in the main period, so the primary experiment
  excludes it. The exclusion is a documented scope decision, not a silent drop.
- **Bundled sample:** every 25th row of days 0–9, in original row order —
  203,095 rows with 179 laundering rows, 18.4 MB. This keeps the time structure
  and the ~0.088% class rate. Sample results check that the pipeline works;
  conclusions about model quality use the full experiment.
- **Split:** time-based by day, fixed in `config.py` — train days 0–5,
  validation days 6–7, test days 8–9.

## Leakage safeguards

- Past features for a transaction use activity from **strictly earlier
  minutes** only; same-minute rows are excluded (`src/features.py`).
- Feature construction never reads `is_laundering`.
- Rule thresholds, the imputer, the one-hot encoder, and the model are all
  fitted on **training days only** (the last two inside a scikit-learn
  `Pipeline`).
- Validation is used for the alert cutoff; the test set is scored once per
  method at the end.

## Repository map

| Path | Purpose |
|---|---|
| `main.py` | Single entry point; `--full` selects the full experiment. |
| `config.py` | Seeds, paths, split days, rule quantiles, model settings. |
| `src/data.py` | Raw CSV loading, cleaning, quality report. |
| `src/features.py` | Past-only transaction and account-history features. |
| `src/rules.py` | Rule baseline with train-only thresholds. |
| `src/metrics.py` | Shared ranking metrics and confusion matrix. |
| `src/split.py` | Time-based split and anomalous-tail exclusion. |
| `src/eda_report.py` | EDA tables/figures for `results/full/`. |
| `src/make_sample.py` | Rebuild the bundled sample from the full CSV. |
| `data/download.py` | Optional Kaggle download helper. |
| `databricks/` | Earlier Spark ingest prototype (not needed for the sample run). |

## Notes for graders

The sample run is a functionality check, so its metrics are intentionally
noisy (only 38 positive test rows). Treat the full-run numbers in
[report.md](report.md) as the evidence for the research question. Variation is
expected from BLAS threading and sklearn version; the seed is fixed at 42.
