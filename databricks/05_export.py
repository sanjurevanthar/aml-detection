# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Export to Parquet
# MAGIC Hand-off to the local ML side (pandas / LightGBM). Two files, joined later on `txn_id`:
# MAGIC - `transactions.parquet`: cleaned transactions + `split` (inputs for `ml/features.py`)
# MAGIC - `rules_scored.parquet`: rule flags + `rule_score` (the baseline's test-set scores)
# MAGIC
# MAGIC Download the folder from the Volume into `data/` on your laptop.

# COMMAND ----------

TXN_TABLE = "workspace.aml.transactions_split"
RULES_TABLE = "workspace.aml.rules_scored"
EXPORT_DIR = "/Volumes/workspace/default/raw_data/export"

# COMMAND ----------

txn = spark.table(TXN_TABLE)
rules = spark.table(RULES_TABLE)

# Sanity checks before writing
n_txn, n_rules = txn.count(), rules.count()
assert n_txn == n_rules, f"row mismatch: {n_txn:,} transactions vs {n_rules:,} rule rows"
assert txn.select("txn_id").distinct().count() == n_txn, "txn_id is not unique"
assert rules.join(txn, "txn_id", "left_anti").count() == 0, "rule rows with no matching transaction"

# COMMAND ----------

# coalesce(1): one file per table. ~5M rows is small enough, and it is easier to download.
(
    txn.coalesce(1)
    .write.mode("overwrite")
    .parquet(f"{EXPORT_DIR}/transactions.parquet")
)
# Drop the label and split here: they already live in transactions.parquet.
(
    rules.drop("is_laundering", "split")
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(f"{EXPORT_DIR}/rules_scored.parquet")
)

# COMMAND ----------

for name in ("transactions", "rules_scored"):
    back = spark.read.parquet(f"{EXPORT_DIR}/{name}.parquet")
    print(f"{name}: {back.count():,} rows, {len(back.columns)} columns")
    assert back.count() == n_txn
display(dbutils.fs.ls(EXPORT_DIR))
