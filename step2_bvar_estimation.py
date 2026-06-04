"""
Step 2: BVAR Estimation for Poland Macro Data
Variables (levels): gdp_yoy, cpi_yoy, wibor_3m
p=2, Minnesota Prior lambda=0.2
Gibbs sampler: 6000 draws, 1000 burn-in, 2 independent chains
Outputs: IRF plots, trace/diagnostics plot, results text
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invwishart

np.random.seed(0)

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df   = pd.read_csv(os.path.join(BASE, "data", "poland_macro_data.csv"), index_col=0)
cols = ["gdp_yoy", "cpi_yoy", "wibor_3m"]
Y_full  = df[cols].values.astype(float)   # (96, 3)
T_data  = len(Y_full)
n, p    = 3, 2
k_lags  = n * p        # 6
k       = k_lags + 1  # 6 lags + intercept = 7
lam     = 0.2
N_TOTAL = 6000
N_BURN  = 1000
N_KEEP  = N_TOTAL - N_BURN   # 5000

print(f"T={T_data}, n={n}, p={p}, k={k}, N_KEEP={N_KEEP}")

# ── 2. Build X and Y_dep ─────────────────────────────────────────────────────
rows_X = []
for t in range(p, T_data):
    lag_block = np.concatenate([Y_full[t - l] for l in range(1, p + 1)])
    rows_X.append(np.append(lag_block, 1.0))   # intercept last

X     = np.array(rows_X)   # (T_e, k)
Y_dep = Y_full[p:]          # (T_e, n)
T_e   = Y_dep.shape[0]
XtX   = X.T @ X
print(f"T_e={T_e}")

# ── 3. AR sigma for Minnesota prior ──────────────────────────────────────────
def ar_sigma_levels(y, p):
    T = len(y)
    Xar = np.column_stack([y[p - l - 1: T - l - 1] for l in range(p)] + [np.ones(T - p)])
    b   = np.linalg.lstsq(Xar, y[p:], rcond=None)[0]
    return np.std(y[p:] - Xar @ b, ddof=p + 1)

sigma_ols = np.array([ar_sigma_levels(Y_full[:, i], p) for i in range(n)])
print(f"AR sigma: {dict(zip(cols, sigma_ols.round(6)))}")

# ── 4. Minnesota prior (random-walk for own first lag, levels) ────────────────
DIFFUSE = 1e6
b_bar   = np.zeros(k * n)
V_d_inv = np.zeros(k * n)

for eq in range(n):
    for lag in range(1, p + 1):
        for var in range(n):
            idx = eq * k + (lag - 1) * n + var
            if var == eq:
                v = (lam / lag) ** 2
                b_bar[idx] = 1.0 if lag == 1 else 0.0
            else:
                v = (lam * sigma_ols[eq] / (lag * sigma_ols[var])) ** 2
                b_bar[idx] = 0.0
            V_d_inv[idx] = 1.0 / v
    # intercept: diffuse
    idx_int          = eq * k + k_lags
    b_bar[idx_int]   = 0.0
    V_d_inv[idx_int] = 1.0 / DIFFUSE

V_prior_inv = np.diag(V_d_inv)
Vb_bar      = V_prior_inv @ b_bar

# IW prior on Sigma
S0  = np.diag(sigma_ols ** 2)
nu0 = n + 1

# ── 5. Gibbs sampler (single chain) ──────────────────────────────────────────
def run_chain(seed, label):
    rng = np.random.default_rng(seed)
    B_draws     = np.zeros((N_KEEP, k, n))
    Sigma_draws = np.zeros((N_KEEP, n, n))
    B_cur       = np.linalg.lstsq(X, Y_dep, rcond=None)[0].copy()
    Sigma_cur   = np.diag(sigma_ols ** 2)

    print(f"  Running {label} …")
    for draw in range(N_TOTAL):
        if draw % 2000 == 0:
            print(f"    {label} draw {draw}/{N_TOTAL}")
        Sig_inv    = np.linalg.inv(Sigma_cur)
        V_post_inv = V_prior_inv + np.kron(Sig_inv, XtX)
        V_post     = np.linalg.inv(V_post_inv)
        b_post     = V_post @ (Vb_bar + np.kron(Sig_inv, X.T) @ Y_dep.ravel(order="F"))
        vec_B      = rng.multivariate_normal(b_post, V_post)
        B_cur      = vec_B.reshape(k, n, order="F")
        E          = Y_dep - X @ B_cur
        S_post     = S0 + E.T @ E
        nu_post    = nu0 + T_e
        Sigma_cur  = invwishart.rvs(df=nu_post, scale=S_post, random_state=rng)
        if draw >= N_BURN:
            s = draw - N_BURN
            B_draws[s]     = B_cur
            Sigma_draws[s] = Sigma_cur
    print(f"    {label} done.")
    return B_draws, Sigma_draws

print("Running Gibbs sampler (2 chains) …")
B1, Sig1 = run_chain(seed=42,  label="Chain 1")
B2, Sig2 = run_chain(seed=123, label="Chain 2")

# Pool chains
B_all     = np.concatenate([B1, B2], axis=0)   # (10000, k, n)
Sigma_all = np.concatenate([Sig1, Sig2], axis=0)

# ── 6. R-hat (Gelman-Rubin) ──────────────────────────────────────────────────
def rhat(chain1, chain2):
    """R-hat for 1-D parameter arrays of equal length."""
    N = len(chain1)
    mean1, mean2 = chain1.mean(), chain2.mean()
    grand_mean   = (mean1 + mean2) / 2.0
    B_var = N * ((mean1 - grand_mean) ** 2 + (mean2 - grand_mean) ** 2)
    W     = (chain1.var(ddof=1) + chain2.var(ddof=1)) / 2.0
    if W < 1e-15:
        return np.nan
    var_hat = (N - 1) / N * W + B_var / N
    return float(np.sqrt(var_hat / W))

# R-hat for all B parameters
rhat_B = np.zeros((k, n))
for eq in range(n):
    for row in range(k):
        rhat_B[row, eq] = rhat(B1[:, row, eq], B2[:, row, eq])

rhat_max = np.nanmax(rhat_B)
rhat_min = np.nanmin(rhat_B)
print(f"\nR-hat B — min: {rhat_min:.4f}, max: {rhat_max:.4f}, mean: {np.nanmean(rhat_B):.4f}")
print(f"Convergence (R-hat < 1.1): {'Yes' if rhat_max < 1.1 else 'No'}")

# ── 7. IRF: wibor_3m shock (index 2) ─────────────────────────────────────────
H          = 20
shock_var  = 2   # wibor_3m
n_draws    = len(B_all)

def compute_irf(B_draws, Sigma_draws, shock_var, resp_var, H):
    irf = np.zeros((len(B_draws), H + 1))
    for d in range(len(B_draws)):
        B   = B_draws[d]       # (k, n)
        Sig = Sigma_draws[d]   # (n, n)
        try:
            P = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            P = np.linalg.cholesky(Sig + 1e-8 * np.eye(n))
        comp = np.zeros((n * p, n * p))
        for lag in range(p):
            comp[:n, lag * n:(lag + 1) * n] = B[lag * n:(lag + 1) * n, :].T
        if p > 1:
            comp[n:, :n * (p - 1)] = np.eye(n * (p - 1))
        state       = np.zeros(n * p)
        state[:n]   = P[:, shock_var]
        irf[d, 0]   = state[resp_var]
        for h in range(1, H + 1):
            state      = comp @ state
            irf[d, h]  = state[resp_var]
    return irf

print("\nComputing IRFs …")
irf_gdp = compute_irf(B_all, Sigma_all, shock_var=2, resp_var=0, H=H)  # → gdp_yoy
irf_cpi = compute_irf(B_all, Sigma_all, shock_var=2, resp_var=1, H=H)  # → cpi_yoy

def credible_bands(irf_draws):
    return {
        "med":  np.median(irf_draws, axis=0),
        "lo68": np.percentile(irf_draws, 16, axis=0),
        "hi68": np.percentile(irf_draws, 84, axis=0),
        "lo90": np.percentile(irf_draws, 5,  axis=0),
        "hi90": np.percentile(irf_draws, 95, axis=0),
    }

cb_gdp = credible_bands(irf_gdp)
cb_cpi = credible_bands(irf_cpi)

# ── 8. IRF plots ──────────────────────────────────────────────────────────────
horizons = np.arange(H + 1)

def plot_irf(cb, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(horizons, cb["lo90"], cb["hi90"], alpha=0.18,
                    color="steelblue", label="90% credible band")
    ax.fill_between(horizons, cb["lo68"], cb["hi68"], alpha=0.38,
                    color="steelblue", label="68% credible band")
    ax.plot(horizons, cb["med"], color="navy", lw=2, label="Posterior median")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close(fig)
    print(f"Saved: outputs/{fname}")

plot_irf(
    cb_gdp,
    "IRF: WIBOR 3M Shock $\\rightarrow$ GDP YoY\n"
    "BVAR(2) · Minnesota Prior ($\\lambda$=0.2) · 2 Chains × 5 000 draws",
    "Response (GDP YoY, %)",
    "irf_rate_gdp.png",
)
plot_irf(
    cb_cpi,
    "IRF: WIBOR 3M Shock $\\rightarrow$ CPI YoY\n"
    "BVAR(2) · Minnesota Prior ($\\lambda$=0.2) · 2 Chains × 5 000 draws",
    "Response (CPI YoY, %)",
    "irf_rate_cpi.png",
)

# ── 9. Trace plot (first 8 parameters, both chains) ──────────────────────────
# First 8 parameters in vec(B) column-major = first 7 of eq0 (gdp_yoy) + first of eq1
param_labels = []
for eq in range(n):
    for lag in range(1, p + 1):
        for var in range(n):
            param_labels.append(f"B[{cols[var]}_L{lag}→{cols[eq]}]")
    param_labels.append(f"B[const→{cols[eq]}]")

# k=7 params per equation; first 8 in vec(B): all 7 of eq0 + first of eq1
param_indices = list(range(8))   # indices into vec(B)

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.ravel()
keep_x = np.arange(N_KEEP)

for idx, ax in zip(param_indices, axes):
    eq  = idx // k
    row = idx % k
    c1  = B1[:, row, eq]
    c2  = B2[:, row, eq]
    ax.plot(keep_x, c1, color="steelblue", lw=0.5, alpha=0.8, label="Chain 1")
    ax.plot(keep_x, c2, color="darkorange",  lw=0.5, alpha=0.8, label="Chain 2")
    ax.set_title(param_labels[idx], fontsize=8)
    ax.set_xlabel("Draw", fontsize=7)
    ax.tick_params(labelsize=7)
    rh = rhat_B[row, eq]
    ax.text(0.98, 0.95, f"R-hat={rh:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            color="green" if rh < 1.1 else "red")

axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle(
    "Trace Plots: First 8 Parameters (2 Chains)\n"
    "BVAR(2) Poland · Minnesota Prior · Gibbs Sampler",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "diagnostics.png"), dpi=150)
plt.close(fig)
print("Saved: outputs/diagnostics.png")

# ── 10. Results text file ─────────────────────────────────────────────────────
B_mean = B_all.mean(0)
B_std  = B_all.std(0)
B_q16  = np.percentile(B_all, 16, 0)
B_q84  = np.percentile(B_all, 84, 0)

reg_names = []
for lag in range(1, p + 1):
    for var in range(n):
        reg_names.append(f"{cols[var]}_L{lag}")
reg_names.append("intercept")

sep  = "=" * 70
dash = "-" * 65

with open(os.path.join(OUTDIR, "step2_results.txt"), "w", encoding="utf-8") as f:
    f.write(f"{sep}\n  Step 2: BVAR Estimation — Poland Macro\n{sep}\n")
    f.write(f"Variables: {cols}\n")
    f.write(f"Lags p={p},  Minnesota prior lambda={lam}  (random-walk prior, levels)\n")
    f.write(f"Chains: 2 x {N_KEEP} kept draws  (burn-in {N_BURN} each)\n")
    f.write(f"Effective observations: T_e={T_e}\n\n")

    # Posterior coefficients
    f.write(f"Posterior Coefficients (pooled chains)\n{dash}\n")
    hdr = f"  {'Regressor':<22} {'Mean':>9} {'Std':>9} {'16%':>9} {'84%':>9}  R-hat\n"
    for eq in range(n):
        f.write(f"\n  Equation: {cols[eq]}\n")
        f.write(hdr)
        f.write("  " + "-" * 60 + "\n")
        for row, name in enumerate(reg_names):
            f.write(
                f"  {name:<22} {B_mean[row,eq]:>9.4f} {B_std[row,eq]:>9.4f}"
                f" {B_q16[row,eq]:>9.4f} {B_q84[row,eq]:>9.4f}  {rhat_B[row,eq]:.4f}\n"
            )

    # R-hat summary
    f.write(f"\n{dash}\n  Gelman-Rubin R-hat Summary\n{dash}\n")
    f.write(f"  Min  R-hat : {rhat_min:.4f}\n")
    f.write(f"  Max  R-hat : {rhat_max:.4f}\n")
    f.write(f"  Mean R-hat : {np.nanmean(rhat_B):.4f}\n")
    f.write(f"  Convergence (all R-hat < 1.1): {'Yes' if rhat_max < 1.1 else 'No'}\n")

    # Posterior Sigma
    Sm = Sigma_all.mean(0)
    f.write(f"\n{dash}\n  Posterior mean of Sigma (residual covariance)\n{dash}\n")
    f.write("  " + "".join(f"{c:>14}" for c in cols) + "\n")
    for i, rl in enumerate(cols):
        f.write("  " + f"{rl:<14}" + "".join(f"{Sm[i,j]:>14.6f}" for j in range(n)) + "\n")

    # IRF tables
    for cb, resp_name, irf_draws in [
        (cb_gdp, "gdp_yoy",  irf_gdp),
        (cb_cpi, "cpi_yoy",  irf_cpi),
    ]:
        f.write(f"\n{dash}\n  IRF: wibor_3m shock -> {resp_name}  (Cholesky)\n{dash}\n")
        f.write(f"  {'h':>3}  {'Median':>10}  {'16%':>10}  {'84%':>10}"
                f"  {'5%':>10}  {'95%':>10}\n")
        f.write("  " + "-" * 58 + "\n")
        for h in range(H + 1):
            f.write(
                f"  {h:>3}  {cb['med'][h]:>10.6f}  {cb['lo68'][h]:>10.6f}"
                f"  {cb['hi68'][h]:>10.6f}  {cb['lo90'][h]:>10.6f}"
                f"  {cb['hi90'][h]:>10.6f}\n"
            )

print("Saved: outputs/step2_results.txt")
print("\nStep 2 complete.")
