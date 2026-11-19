# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Time-based split
# MAGIC Add a `split` column (train / val / test, ~60 / 20 / 20) to the transactions.
# MAGIC This is done ONCE and saved. Rules and ML models both read this table, so they
# MAGIC are always tested on exactly the same rows.
# MAGIC
# MAGIC **Why by time, not random?** In real life we predict the future from the past.
# MAGIC A random split would let the model see future behavior during training.

# COMMAND ----------

from pyspark.sql import functions as F

SRC_TABLE = "workspace.aml.transactions"
OUT_TABLE = "workspace.aml.transactions_split"

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20  # test gets the remaining 0.20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find the cut timestamps
# MAGIC Cut by **row percentile** of time (not by calendar days), because the last
# MAGIC days of the data are sparse. Rows with the same timestamp always land in the
# MAGIC same split, so no minute is ever cut in half.

# COMMAND ----------

df = spark.table(SRC_TABLE).withColumn("ts_unix", F.col("timestamp").cast("long"))

t_train_end, t_val_end = df.approxQuantile(
    "ts_unix", [TRAIN_FRAC, TRAIN_FRAC + VAL_FRAC], 0.0001
)
print("train/val cut (unix):", t_train_end)
print("val/test cut  (unix):", t_val_end)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Assign the split
# MAGIC - `ts <= t_train_end` -> train
# MAGIC - `ts <= t_val_end`   -> val
# MAGIC - otherwise           -> test

# COMMAND ----------

df = df.withColumn(
    "split",
    F.when(F.col("ts_unix") <= t_train_end, "train")
    .when(F.col("ts_unix") <= t_val_end, "val")
    .otherwise("test"),
).drop("ts_unix")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save (one source of truth)

# COMMAND ----------

df.write.mode("overwrite").format("delta").saveAsTable(OUT_TABLE)
print(f"Saved {OUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sanity checks
# MAGIC - Splits should be roughly 60 / 20 / 20.
# MAGIC - Time ranges must NOT overlap, and must go train < val < test.
# MAGIC - Every split needs enough laundering cases to measure anything.
# MAGIC   (If val or test has only a handful, metrics like recall@top 1% become noisy.)

# COMMAND ----------

t = spark.table(OUT_TABLE)
total = t.count()

summary = (
    t.groupBy("split")
    .agg(
        F.count("*").alias("rows"),
        F.sum("is_laundering").alias("laundering"),
        F.min("timestamp").alias("first_ts"),
        F.max("timestamp").alias("last_ts"),
    )
    .withColumn("pct_of_rows", F.round(F.col("rows") / total * 100, 1))
    .withColumn("laundering_rate_pct", F.round(F.col("laundering") / F.col("rows") * 100, 4))
    .orderBy("first_ts")
)
display(summary)

# COMMAND ----------

# No row lost, no txn_id duplicated
assert total == spark.table(SRC_TABLE).count(), "row count changed!"
assert t.select("txn_id").distinct().count() == total, "duplicate txn_id!"
assert t.filter("split IS NULL").count() == 0, "some rows have no split!"
print("OK: all checks passed")
