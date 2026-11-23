# src/eda_report.py -- saves EDA tables and figures to results/full/
# Run from the project root:  python -m src.eda_report
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_raw, quality_report, clean

OUT = "results/full"
os.makedirs(OUT, exist_ok=True)

df = load_raw("data/raw/HI-Small_Trans.csv")
report = quality_report(df)
df = clean(df)

# ---- extra checks ----
key = [c for c in df.columns if c not in ("row_id", "day")]
dups = df[df.duplicated(subset=key, keep=False)]
print("\nRows inside duplicate groups:", len(dups),
      "| laundering among them:", int(dups["is_laundering"].sum()))

same_acct = df["src_acct"] == df["dst_acct"]
same_acct_bank = same_acct & (df["src_bank"] == df["dst_bank"])
print("Same account ID:", int(same_acct.sum()),
      "| same account ID AND same bank:", int(same_acct_bank.sum()))
print("Laundering among self-transfers:", int(df.loc[same_acct, "is_laundering"].sum()))
print(pd.crosstab(df["pay_format"], same_acct))

print("\nTop payment currencies:")
print(df["pay_ccy"].value_counts().head(10))
print("Currency mismatch share:", float((df["pay_ccy"] != df["recv_ccy"]).mean()))

# ---- tables ----
by_day = df.groupby("day")["is_laundering"].agg(transactions="count", laundering="sum")
by_day["rate"] = by_day["laundering"] / by_day["transactions"]
by_fmt = df.groupby("pay_format")["is_laundering"].agg(transactions="count", laundering="sum")
by_fmt["rate"] = by_fmt["laundering"] / by_fmt["transactions"]

pd.Series(report).to_csv(f"{OUT}/data_summary.csv", header=["value"])
by_day.to_csv(f"{OUT}/eda_by_day.csv")
by_fmt.to_csv(f"{OUT}/eda_by_pay_format.csv")

# ---- Figure 1: per day ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
by_day["transactions"].plot.bar(ax=ax[0], logy=True, color="tab:blue")
ax[0].set_title("Transactions per day (log scale)")
ax[0].set_xlabel("Day index (0 = 2022-09-01)")
ax[0].set_ylabel("Number of transactions")
by_day["laundering"].plot.bar(ax=ax[1], color="tab:red")
ax[1].set_title("Laundering transactions per day")
ax[1].set_xlabel("Day index (0 = 2022-09-01)")
ax[1].set_ylabel("Number of laundering transactions")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_by_day.png", dpi=150)
plt.close(fig)

# ---- Figure 2: amount distribution ----
la = np.log10(df.loc[df["is_laundering"] == 1, "amt_paid"])
no = np.log10(df.loc[df["is_laundering"] == 0, "amt_paid"])
bins = np.linspace(min(la.min(), no.min()), max(la.max(), no.max()), 80)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(no, bins=bins, alpha=0.5, density=True, label="normal")
ax.hist(la, bins=bins, alpha=0.5, density=True, label="laundering")
ax.set_title("Amount paid (log10 scale), normal vs laundering")
ax.set_xlabel("log10(amount paid, in the payment currency)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eda_amount_dist.png", dpi=150)
plt.close(fig)

# ---- Figure 3: laundering rate by payment format ----
fig, ax = plt.subplots(figsize=(8, 4))
(by_fmt["rate"] * 100).plot.bar(ax=ax, color="tab:orange")
ax.set_title("Laundering rate by payment format")
ax.set_xlabel("Payment format")
ax.set_ylabel("Laundering rate (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/eda_by_pay_format.png", dpi=150)
plt.close(fig)

print("\nSaved files in", OUT)
print(sorted(os.listdir(OUT)))
