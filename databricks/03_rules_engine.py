# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Rule engine (the baseline detector)
# MAGIC
# MAGIC A classic bank-style monitoring system: each rule is a simple yes/no flag that
# MAGIC mirrors a red flag from AML guidance. `rule_score` = weighted sum of the flags.
# MAGIC A higher score means "look at this one first".
# MAGIC
# MAGIC **Important design rules**
# MAGIC - The rules NEVER look at `is_laundering`. The label is only used afterwards to
# MAGIC   judge them, exactly like the ML models.
# MAGIC - Any threshold learned from data (99th percentiles) uses TRAIN rows only.
# MAGIC - Weights are set by judgment (severity), not fitted to the labels. Fitting weights
# MAGIC   to the labels would turn the rules into a model and make the comparison unfair.
# MAGIC
# MAGIC ## Rule catalogue
# MAGIC | Flag | Rule | Basis |
# MAGIC |---|---|---|
# MAGIC | `flag_large_amount` | amount above the 99th percentile for its currency (train only) | Threshold rules (large-value transactions) |
# MAGIC | `flag_near_threshold` | USD amount from 9,000 up to 9,999.99 | Structuring: staying under the $10,000 US reporting threshold |
# MAGIC | `flag_round_amount` | amount is a multiple of 1,000 | Round amounts are a common behavioral signal |
# MAGIC | `flag_currency_change` | paid currency differs from received currency | Cross-currency movement |
# MAGIC | `flag_cash_or_bitcoin` | payment format is Cash or Bitcoin | Higher-risk, harder-to-trace channels |
# MAGIC | `flag_high_velocity` | more than 10 transfers from the same account in 24h | Velocity rule |
# MAGIC | `flag_fan_out` | unusually many distinct receivers in 24h (above train 99th pct) | Dispersal of funds |
# MAGIC | `flag_fan_in` | unusually many distinct senders into one account in 24h (above train 99th pct) | Funnel / collection account |
# MAGIC | `flag_pass_through` | account sends out 90-110% of what it received in the past 24h (same currency) | Rapid movement of funds / layering |
# MAGIC
# MAGIC Not covered (needs graph analysis, which is what ML features are for): cycles,
# MAGIC bipartite and stack patterns spanning many accounts.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

SRC_TABLE = "workspace.aml.transactions_split"
OUT_TABLE = "workspace.aml.rules_scored"

DAY = 86400  # seconds

# Severity weights, set by judgment (NOT fitted to the labels).
WEIGHTS = {
    "flag_large_amount": 2,
    "flag_near_threshold": 2,
    "flag_round_amount": 1,
    "flag_currency_change": 1,
    "flag_cash_or_bitcoin": 2,
    "flag_high_velocity": 2,
    "flag_fan_out": 2,
    "flag_fan_in": 2,
    "flag_pass_through": 3,
}

VELOCITY_LIMIT = 10  # transfers in 24h
ROUND_BASE = 1000
NEAR_THRESHOLD_LOW, NEAR_THRESHOLD_HIGH = 9000, 10000  # USD
PASS_LOW, PASS_HIGH = 0.9, 1.1  # outflow / inflow band

# COMMAND ----------

df = spark.table(SRC_TABLE).withColumn("ts_unix", F.col("timestamp").cast("long"))

# Quick look at the categorical values the rules rely on
display(df.groupBy("payment_format").count().orderBy(F.desc("count")))
display(df.groupBy("payment_currency").count().orderBy(F.desc("count")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Account behavior over the past 24 hours
# MAGIC Window per account, ordered by time, covering the last 24h (86,400 seconds).
# MAGIC Note: with a time-range window, rows in the same minute count as peers, so a
# MAGIC transaction can "see" others from its own minute. That is acceptable for rules;
# MAGIC remember it when building ML features.

# COMMAND ----------

w_from_24h = Window.partitionBy("from_key").orderBy("ts_unix").rangeBetween(-DAY, 0)
w_to_24h = Window.partitionBy("to_key").orderBy("ts_unix").rangeBetween(-DAY, 0)

df = (
    df.withColumn("velocity_24h", F.count("*").over(w_from_24h))
    .withColumn("uniq_receivers_24h", F.size(F.collect_set("to_key").over(w_from_24h)))
    .withColumn("uniq_senders_24h", F.size(F.collect_set("from_key").over(w_to_24h)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pass-through: money in, then ~same money out
# MAGIC For every account and currency, build one stream of events: money received (in)
# MAGIC and money sent (out). For each "out" event, sum what the same account received
# MAGIC in the 24h BEFORE it (strictly earlier minutes, so no peeking at the present).

# COMMAND ----------

outgoing = df.select(
    "txn_id",
    F.col("from_key").alias("acct"),
    F.col("payment_currency").alias("ccy"),
    "ts_unix",
    F.lit(0.0).alias("in_amt"),
    F.lit(True).alias("is_out"),
)
incoming = df.select(
    "txn_id",
    F.col("to_key").alias("acct"),
    F.col("receiving_currency").alias("ccy"),
    "ts_unix",
    F.col("amount_received").alias("in_amt"),
    F.lit(False).alias("is_out"),
)

w_events = Window.partitionBy("acct", "ccy").orderBy("ts_unix").rangeBetween(-DAY, -1)
received_prev = (
    outgoing.unionByName(incoming)
    .withColumn("received_prev_24h", F.sum("in_amt").over(w_events))
    .filter("is_out")
    .select("txn_id", F.coalesce("received_prev_24h", F.lit(0.0)).alias("received_prev_24h"))
)

df = df.join(received_prev, "txn_id", "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Thresholds learned on TRAIN rows only

# COMMAND ----------

train = df.filter("split = 'train'")

amount_p99 = train.groupBy("payment_currency").agg(
    F.percentile_approx("amount_paid", 0.99).alias("amount_p99")
)
df = df.join(F.broadcast(amount_p99), "payment_currency", "left")

fan_out_limit = train.approxQuantile("uniq_receivers_24h", [0.99], 0.001)[0]
fan_in_limit = train.approxQuantile("uniq_senders_24h", [0.99], 0.001)[0]
print(f"fan-out limit (train 99th pct, distinct receivers/24h): {fan_out_limit}")
print(f"fan-in limit  (train 99th pct, distinct senders/24h):   {fan_in_limit}")
display(amount_p99.orderBy("payment_currency"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Flags and rule_score

# COMMAND ----------

df = (
    df.withColumn("flag_large_amount", F.coalesce(F.col("amount_paid") > F.col("amount_p99"), F.lit(False)))
    .withColumn(
        "flag_near_threshold",
        (F.col("payment_currency") == "US Dollar")
        & (F.col("amount_paid") >= NEAR_THRESHOLD_LOW)
        & (F.col("amount_paid") < NEAR_THRESHOLD_HIGH),
    )
    .withColumn(
        "flag_round_amount",
        (F.col("amount_paid") >= ROUND_BASE) & (F.col("amount_paid") % ROUND_BASE == 0),
    )
    .withColumn("flag_currency_change", F.col("payment_currency") != F.col("receiving_currency"))
    .withColumn("flag_cash_or_bitcoin", F.col("payment_format").isin("Cash", "Bitcoin"))
    .withColumn("flag_high_velocity", F.col("velocity_24h") > VELOCITY_LIMIT)
    .withColumn("flag_fan_out", F.col("uniq_receivers_24h") > fan_out_limit)
    .withColumn("flag_fan_in", F.col("uniq_senders_24h") > fan_in_limit)
    .withColumn(
        "flag_pass_through",
        (F.col("received_prev_24h") > 0)
        & (F.col("amount_paid") >= PASS_LOW * F.col("received_prev_24h"))
        & (F.col("amount_paid") <= PASS_HIGH * F.col("received_prev_24h")),
    )
)

flag_cols = list(WEIGHTS)
df = df.withColumn(
    "rule_score",
    sum(F.col(c).cast("int") * w for c, w in WEIGHTS.items()),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save (flags + supporting numbers; the label is kept only for judging)

# COMMAND ----------

scored = df.select(
    "txn_id",
    "split",
    "is_laundering",
    "velocity_24h",
    "uniq_receivers_24h",
    "uniq_senders_24h",
    "received_prev_24h",
    *flag_cols,
    "rule_score",
)
scored.write.mode("overwrite").format("delta").saveAsTable(OUT_TABLE)
print(f"Saved {OUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. How good is each rule? (train and val only; test stays sealed)
# MAGIC - **flag rate**: share of transactions the rule flags
# MAGIC - **precision**: of flagged transactions, share that are really laundering
# MAGIC - **recall**: of all laundering, share the rule catches
# MAGIC - **lift**: precision divided by the overall laundering rate (1.0 = no better than random)

# COMMAND ----------

scored = spark.table(OUT_TABLE)

aggs = [F.count("*").alias("rows"), F.sum("is_laundering").alias("laundering")]
for c in flag_cols:
    aggs.append(F.sum(F.col(c).cast("int")).alias(f"{c}__flagged"))
    aggs.append(F.sum(F.when(F.col(c) & (F.col("is_laundering") == 1), 1).otherwise(0)).alias(f"{c}__tp"))

import pandas as pd

records = []
for row in scored.filter("split IN ('train', 'val')").groupBy("split").agg(*aggs).collect():
    base_rate = row["laundering"] / row["rows"]
    for c in flag_cols:
        flagged, tp = row[f"{c}__flagged"], row[f"{c}__tp"]
        precision = tp / flagged if flagged else 0.0
        records.append(
            {
                "split": row["split"],
                "rule": c,
                "flag_rate_pct": round(flagged / row["rows"] * 100, 3),
                "precision_pct": round(precision * 100, 3),
                "recall_pct": round(tp / row["laundering"] * 100, 2),
                "lift": round(precision / base_rate, 2),
            }
        )

display(pd.DataFrame(records).sort_values(["split", "rule"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. The combined rule_score on VALIDATION: top 1% review budget
# MAGIC Investigators review only the 1% highest-scored transactions. Many rows share
# MAGIC the same integer score, so ties are broken by a hash of `txn_id` (arbitrary, and it
# MAGIC never uses the label). This gives the 2x test a fixed, fair review budget.


# COMMAND ----------


def top_pct_report(scored_df, split_name, pct=0.01):
    s = scored_df.filter(F.col("split") == split_name)
    n = s.count()
    positives = s.agg(F.sum("is_laundering")).first()[0]
    k = int(n * pct)
    # Global sort + limit (no un-partitioned window, which forced one partition).
    # Only k label values (~1% of rows) come back to the driver, already in rank order.
    top_labels = (
        s.orderBy(F.col("rule_score").desc(), F.xxhash64("txn_id"))
        .limit(k)
        .select("is_laundering")
        .toPandas()["is_laundering"]
        .to_numpy()
    )
    tp = int(top_labels.sum())
    tp100 = int(top_labels[:100].sum())
    fp, fn = k - tp, positives - tp
    tn = n - k - fn
    print(f"[{split_name}] review budget: top {pct:.0%} = {k:,} of {n:,} transactions")
    print(f"  laundering in split:  {positives:,}")
    print(f"  recall@top{pct:.0%}:        {tp / positives:.2%}  ({tp:,} caught)")
    print(f"  precision@top{pct:.0%}:     {tp / k:.2%}")
    print(f"  precision@top100:     {tp100 / 100:.2%}")
    print(f"  confusion matrix:  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    return {"recall_at_1pct": tp / positives}


val_report = top_pct_report(scored, "val")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Score distribution (is there enough spread to rank?)

# COMMAND ----------

display(
    scored.filter("split = 'val'")
    .groupBy("rule_score")
    .agg(F.count("*").alias("rows"), F.sum("is_laundering").alias("laundering"))
    .withColumn("laundering_rate_pct", F.round(F.col("laundering") / F.col("rows") * 100, 4))
    .orderBy("rule_score")
)
