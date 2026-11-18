# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest
# MAGIC Read the raw IBM AML CSV, clean the column names, build keys and `txn_id`,
# MAGIC and save a Delta table `workspace.aml.transactions`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

RAW_PATH = "/Volumes/workspace/default/raw_data/HI-Small_Trans.csv"
SCHEMA = "workspace.aml"
TABLE = f"{SCHEMA}.transactions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the raw CSV
# MAGIC Everything is read as text first (no schema guessing). Bank ids like `010`
# MAGIC have leading zeros that must NOT be lost, so they stay strings.

# COMMAND ----------

raw = spark.read.csv(RAW_PATH, header=True, inferSchema=False)
print("Raw columns:", raw.columns)  # note: "Account" appears twice

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Rename columns BY POSITION
# MAGIC The header has two columns both called "Account", so renaming by name is
# MAGIC ambiguous. `toDF` renames by position, which is safe.

# COMMAND ----------

NEW_NAMES = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
assert len(raw.columns) == len(NEW_NAMES), f"Expected {len(NEW_NAMES)} columns, got {len(raw.columns)}"
df = raw.toDF(*NEW_NAMES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fix types and build keys
# MAGIC - `timestamp`: text -> real timestamp (format `2022/09/01 00:20`)
# MAGIC - amounts -> double, `is_laundering` -> int (0/1)
# MAGIC - `from_key` / `to_key` = `bank_account`. The same account number can exist at
# MAGIC   two banks, so bank + account is the globally unique id.

# COMMAND ----------

df = (
    df.withColumn("timestamp", F.to_timestamp("timestamp", "yyyy/MM/dd HH:mm"))
    .withColumn("amount_received", F.col("amount_received").cast("double"))
    .withColumn("amount_paid", F.col("amount_paid").cast("double"))
    .withColumn("is_laundering", F.col("is_laundering").cast("int"))
    .withColumn("from_key", F.concat_ws("_", "from_bank", "from_account"))
    .withColumn("to_key", F.concat_ws("_", "to_bank", "to_account"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build `txn_id`
# MAGIC A unique number per transaction, ordered by time. Many rows share the same
# MAGIC minute, so extra columns break ties to make the order deterministic (the same
# MAGIC every run). Note: a window with no `partitionBy` runs on one partition. That is
# MAGIC fine for 5M rows, but it is why this step is slower than the others.

# COMMAND ----------

order_cols = [
    "timestamp",
    "from_key",
    "to_key",
    "amount_paid",
    "amount_received",
    "payment_currency",
    "receiving_currency",
    "payment_format",
    "is_laundering",
]
w = Window.orderBy(*order_cols)
df = df.withColumn("txn_id", F.row_number().over(w))

# Put txn_id first for readability
df = df.select("txn_id", *[c for c in df.columns if c != "txn_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save as a Delta table

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
df.write.mode("overwrite").format("delta").saveAsTable(TABLE)
print(f"Saved {TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sanity checks
# MAGIC Expect: ~5,078,345 rows, laundering rate ~0.1%, no null timestamps or
# MAGIC amounts, `txn_id` unique.

# COMMAND ----------

t = spark.table(TABLE)

n = t.count()
n_laundering = t.filter("is_laundering = 1").count()
print(f"Rows: {n:,}")
print(f"Laundering: {n_laundering:,} ({n_laundering / n:.4%})")
print(f"Distinct txn_id: {t.select('txn_id').distinct().count():,} (must equal rows)")
print(f"Time range: {t.agg(F.min('timestamp'), F.max('timestamp')).first()}")

# COMMAND ----------

# Null counts per column (all should be 0)
t.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in t.columns]).show(vertical=True)

# COMMAND ----------

display(t.orderBy("txn_id").limit(10))
