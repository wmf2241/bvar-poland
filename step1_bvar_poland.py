"""
Step 1: Data diagnostics for Polish BVAR
- ADF unit-root tests (levels + first differences)
- Johansen cointegration test
- VAR lag-order selection
- Time-series plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "poland_macro_data.csv"), index_col=0)
df.index.name = "quarter"
cols      = ["gdp_yoy", "cpi_yoy", "wibor_3m"]
col_labels = {"gdp_yoy": "GDP YoY (%)", "cpi_yoy": "CPI YoY (%)", "wibor_3m": "WIBOR 3M (%)"}
df = df[cols]

print("=" * 65)
print("  Step 1: Poland BVAR Data Diagnostics")
print("=" * 65)
print(f"Observations: {len(df)}  ({df.index[0]} to {df.index[-1]})")
print("\nDescriptive statistics:")
print(df.describe().round(4).to_string())

# ── 2. ADF unit-root tests ────────────────────────────────────────────────────
def run_adf(series, label, maxlag=4):
    """Return ADF result dict for a single series."""
    result = adfuller(series.dropna(), maxlag=maxlag, autolag="AIC", regression="c")
    return {
        "variable"  : label,
        "ADF stat"  : round(result[0], 4),
        "p-value"   : round(result[1], 4),
        "lags used" : result[2],
        "1%"        : round(result[4]["1%"], 3),
        "5%"        : round(result[4]["5%"], 3),
        "10%"       : round(result[4]["10%"], 3),
        "stationary": "Yes" if result[1] < 0.05 else "No",
    }

adf_levels = [run_adf(df[c], c) for c in cols]
adf_diffs  = [run_adf(df[c].diff().dropna(), f"Δ{c}") for c in cols]

adf_df = pd.DataFrame(adf_levels + adf_diffs)
print("\n\nADF Unit-Root Tests")
print("-" * 65)
print(adf_df.to_string(index=False))

# ── 3. Johansen cointegration test ───────────────────────────────────────────
print("\n\nJohansen Cointegration Test  (det_order=0, k_ar_diff=2)")
print("-" * 65)
jres = coint_johansen(df[cols].dropna(), det_order=0, k_ar_diff=2)

# Trace statistic
trace_cv = jres.cvt          # (n_r, 3) critical values at 90%, 95%, 99%
trace_st = jres.lr1          # trace statistics
eig_st   = jres.lr2          # max-eigenvalue statistics
eig_cv   = jres.cvm

print("\nTrace test (H0: at most r cointegrating vectors):")
print(f"  {'r':<4} {'Trace stat':>12} {'CV 10%':>8} {'CV 5%':>8} {'CV 1%':>8}  Reject H0 @ 5%?")
print("  " + "-" * 55)
for r in range(len(cols)):
    reject = "Yes" if trace_st[r] > trace_cv[r, 1] else "No"
    print(f"  {r:<4} {trace_st[r]:>12.4f} {trace_cv[r,0]:>8.3f} {trace_cv[r,1]:>8.3f} {trace_cv[r,2]:>8.3f}  {reject}")

print("\nMax-Eigenvalue test (H0: r cointegrating vectors vs r+1):")
print(f"  {'r':<4} {'Max-Eig stat':>12} {'CV 10%':>8} {'CV 5%':>8} {'CV 1%':>8}  Reject H0 @ 5%?")
print("  " + "-" * 55)
for r in range(len(cols)):
    reject = "Yes" if eig_st[r] > eig_cv[r, 1] else "No"
    print(f"  {r:<4} {eig_st[r]:>12.4f} {eig_cv[r,0]:>8.3f} {eig_cv[r,1]:>8.3f} {eig_cv[r,2]:>8.3f}  {reject}")

n_coint = sum(trace_st[r] > trace_cv[r, 1] for r in range(len(cols)))
print(f"\nConclusion (trace test @ 5%): {n_coint} cointegrating vector(s) found.")

# ── 4. VAR lag-order selection ────────────────────────────────────────────────
print("\n\nVAR Lag-Order Selection  (maxlags=6)")
print("-" * 65)
model    = VAR(df[cols].dropna())
lag_sel  = model.select_order(maxlags=6)
ic_df    = pd.DataFrame({
    "AIC" : lag_sel.ics["aic"],
    "BIC" : lag_sel.ics["bic"],
    "HQIC": lag_sel.ics["hqic"],
})
ic_df.index.name = "lag"
print(ic_df.round(4).to_string())

p_aic  = int(lag_sel.aic)
p_bic  = int(lag_sel.bic)
p_hqic = int(lag_sel.hqic)
print(f"\nRecommended lags — AIC: {p_aic},  BIC: {p_bic},  HQIC: {p_hqic}")

# ── 5. Time-series plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
colors = ["steelblue", "darkorange", "forestgreen"]
titles = [
    "GDP YoY Growth Rate (%)",
    "HICP CPI YoY Growth Rate (%)",
    "WIBOR 3M Interest Rate (%)",
]

x = np.arange(len(df))
for ax, col, color, title in zip(axes, cols, colors, titles):
    ax.plot(x, df[col].values, color=color, lw=1.6)
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.3)

# x-axis ticks every 4 quarters (1 year)
tick_pos   = list(range(0, len(df), 4))
tick_labels = [df.index[i] for i in tick_pos]
axes[-1].set_xticks(tick_pos)
axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

fig.suptitle(
    "Poland Macroeconomic Data  (2000-Q1 to 2023-Q4)\n"
    "Source: Eurostat",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "poland_data_plot.png"), dpi=150)
plt.close(fig)
print("\nSaved: outputs/poland_data_plot.png")

# ── 6. Write results to text file ────────────────────────────────────────────
with open(os.path.join(OUTDIR, "poland_step1_results.txt"), "w", encoding="utf-8") as f:
    sep  = "=" * 65
    dash = "-" * 65

    f.write(f"{sep}\n  Step 1: Poland BVAR Data Diagnostics\n{sep}\n")
    f.write(f"Sample: {df.index[0]} to {df.index[-1]}  (N={len(df)})\n")
    f.write(f"Variables: {cols}\n\n")

    f.write(f"Descriptive Statistics\n{dash}\n")
    f.write(df.describe().round(4).to_string())
    f.write("\n\n")

    f.write(f"ADF Unit-Root Tests\n{dash}\n")
    f.write("Levels:\n")
    hdr = f"  {'Variable':<14} {'ADF stat':>10} {'p-value':>8} {'Lags':>5} {'1%':>7} {'5%':>7} {'10%':>7}  Stationary?\n"
    f.write(hdr)
    f.write("  " + "-" * 62 + "\n")
    for row in adf_levels:
        f.write(
            f"  {row['variable']:<14} {row['ADF stat']:>10.4f} {row['p-value']:>8.4f}"
            f" {row['lags used']:>5}  {row['1%']:>6}  {row['5%']:>6}  {row['10%']:>6}"
            f"  {row['stationary']}\n"
        )
    f.write("\nFirst Differences:\n")
    f.write(hdr)
    f.write("  " + "-" * 62 + "\n")
    for row in adf_diffs:
        f.write(
            f"  {row['variable']:<14} {row['ADF stat']:>10.4f} {row['p-value']:>8.4f}"
            f" {row['lags used']:>5}  {row['1%']:>6}  {row['5%']:>6}  {row['10%']:>6}"
            f"  {row['stationary']}\n"
        )

    f.write(f"\nJohansen Cointegration Test  (det_order=0, k_ar_diff=2)\n{dash}\n")
    f.write("Trace test (H0: at most r cointegrating vectors):\n")
    f.write(f"  {'r':<4} {'Trace stat':>12} {'CV 10%':>8} {'CV 5%':>8} {'CV 1%':>8}  Reject H0 @ 5%?\n")
    f.write("  " + "-" * 55 + "\n")
    for r in range(len(cols)):
        reject = "Yes" if trace_st[r] > trace_cv[r, 1] else "No"
        f.write(
            f"  {r:<4} {trace_st[r]:>12.4f} {trace_cv[r,0]:>8.3f}"
            f" {trace_cv[r,1]:>8.3f} {trace_cv[r,2]:>8.3f}  {reject}\n"
        )
    f.write("\nMax-Eigenvalue test:\n")
    f.write(f"  {'r':<4} {'Max-Eig stat':>12} {'CV 10%':>8} {'CV 5%':>8} {'CV 1%':>8}  Reject H0 @ 5%?\n")
    f.write("  " + "-" * 55 + "\n")
    for r in range(len(cols)):
        reject = "Yes" if eig_st[r] > eig_cv[r, 1] else "No"
        f.write(
            f"  {r:<4} {eig_st[r]:>12.4f} {eig_cv[r,0]:>8.3f}"
            f" {eig_cv[r,1]:>8.3f} {eig_cv[r,2]:>8.3f}  {reject}\n"
        )
    f.write(f"\nConclusion (trace @ 5%): {n_coint} cointegrating vector(s) found.\n")

    f.write(f"\nVAR Lag-Order Selection  (maxlags=6)\n{dash}\n")
    f.write(ic_df.round(4).to_string())
    f.write(f"\n\nRecommended lags — AIC: {p_aic},  BIC: {p_bic},  HQIC: {p_hqic}\n")

print("Saved: outputs/poland_step1_results.txt")
print("\nStep 1 complete.")
