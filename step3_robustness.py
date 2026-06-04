"""
Step 3: Robustness — Minnesota Prior Sensitivity
Compare lambda=0.1 (tight), 0.2 (baseline), 0.4 (loose)
IRF: wibor_3m shock -> cpi_yoy, all three on one plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invwishart

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df   = pd.read_csv(os.path.join(BASE, "data", "poland_macro_data.csv"), index_col=0)
cols = ["gdp_yoy", "cpi_yoy", "wibor_3m"]
Y_full  = df[cols].values.astype(float)
T_data  = len(Y_full)
n, p    = 3, 2
k_lags  = n * p
k       = k_lags + 1      # 6 lags + intercept
N_TOTAL = 6000
N_BURN  = 1000
N_KEEP  = N_TOTAL - N_BURN
H       = 20

# ── 2. Build X and Y_dep ─────────────────────────────────────────────────────
rows_X = []
for t in range(p, T_data):
    lag_block = np.concatenate([Y_full[t - l] for l in range(1, p + 1)])
    rows_X.append(np.append(lag_block, 1.0))

X     = np.array(rows_X)
Y_dep = Y_full[p:]
T_e   = Y_dep.shape[0]
XtX   = X.T @ X

# AR sigma (levels) for Minnesota scaling
def ar_sigma_levels(y, p):
    T = len(y)
    Xar = np.column_stack([y[p-l-1:T-l-1] for l in range(p)] + [np.ones(T-p)])
    b   = np.linalg.lstsq(Xar, y[p:], rcond=None)[0]
    return np.std(y[p:] - Xar @ b, ddof=p+1)

sigma_ols = np.array([ar_sigma_levels(Y_full[:, i], p) for i in range(n)])

S0  = np.diag(sigma_ols ** 2)
nu0 = n + 1

# ── 3. Build prior for a given lambda ────────────────────────────────────────
DIFFUSE = 1e6

def build_prior(lam):
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
        idx_int          = eq * k + k_lags
        b_bar[idx_int]   = 0.0
        V_d_inv[idx_int] = 1.0 / DIFFUSE
    V_prior_inv = np.diag(V_d_inv)
    return V_prior_inv, V_prior_inv @ b_bar

# ── 4. Gibbs sampler ──────────────────────────────────────────────────────────
def run_gibbs(lam, seed=42):
    V_prior_inv, Vb_bar = build_prior(lam)
    rng     = np.random.default_rng(seed)
    B_draws = np.zeros((N_KEEP, k, n))
    Sig_draws = np.zeros((N_KEEP, n, n))
    B_cur   = np.linalg.lstsq(X, Y_dep, rcond=None)[0].copy()
    Sig_cur = np.diag(sigma_ols ** 2)

    for draw in range(N_TOTAL):
        Sig_inv    = np.linalg.inv(Sig_cur)
        V_post_inv = V_prior_inv + np.kron(Sig_inv, XtX)
        V_post     = np.linalg.inv(V_post_inv)
        b_post     = V_post @ (Vb_bar + np.kron(Sig_inv, X.T) @ Y_dep.ravel(order="F"))
        vec_B      = rng.multivariate_normal(b_post, V_post)
        B_cur      = vec_B.reshape(k, n, order="F")
        E          = Y_dep - X @ B_cur
        Sig_cur    = invwishart.rvs(df=nu0 + T_e, scale=S0 + E.T @ E, random_state=rng)
        if draw >= N_BURN:
            s = draw - N_BURN
            B_draws[s]   = B_cur
            Sig_draws[s] = Sig_cur
    return B_draws, Sig_draws

# ── 5. IRF: wibor_3m (index 2) -> cpi_yoy (index 1) ─────────────────────────
def compute_irf(B_draws, Sig_draws, shock_var, resp_var, H):
    irf = np.zeros((len(B_draws), H + 1))
    for d in range(len(B_draws)):
        B   = B_draws[d]
        Sig = Sig_draws[d]
        try:
            P = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            P = np.linalg.cholesky(Sig + 1e-8 * np.eye(n))
        comp = np.zeros((n * p, n * p))
        for lag in range(p):
            comp[:n, lag*n:(lag+1)*n] = B[lag*n:(lag+1)*n, :].T
        if p > 1:
            comp[n:, :n*(p-1)] = np.eye(n*(p-1))
        state       = np.zeros(n * p)
        state[:n]   = P[:, shock_var]
        irf[d, 0]   = state[resp_var]
        for h in range(1, H + 1):
            state     = comp @ state
            irf[d, h] = state[resp_var]
    return irf

def credible_bands(irf_draws):
    return {
        "med":  np.median(irf_draws, axis=0),
        "lo68": np.percentile(irf_draws, 16, axis=0),
        "hi68": np.percentile(irf_draws, 84, axis=0),
        "lo90": np.percentile(irf_draws, 5,  axis=0),
        "hi90": np.percentile(irf_draws, 95, axis=0),
    }

# ── 6. Run all three models ───────────────────────────────────────────────────
lambdas = [0.1, 0.2, 0.4]
results = {}

for lam in lambdas:
    print(f"Running BVAR with lambda={lam} …")
    B_d, S_d = run_gibbs(lam, seed=42)
    irf_d    = compute_irf(B_d, S_d, shock_var=2, resp_var=1, H=H)
    results[lam] = {
        "B_draws":   B_d,
        "Sig_draws": S_d,
        "irf":       irf_d,
        "cb":        credible_bands(irf_d),
    }
    print(f"  done. Median IRF peak: {results[lam]['cb']['med'].min():.4f}")

# ── 7. Robustness comparison plot ─────────────────────────────────────────────
horizons = np.arange(H + 1)
styles = {
    0.1: {"color": "firebrick",  "label": r"$\lambda=0.1$ (tight prior)"},
    0.2: {"color": "navy",       "label": r"$\lambda=0.2$ (baseline)"},
    0.4: {"color": "forestgreen","label": r"$\lambda=0.4$ (loose prior)"},
}
fill_alpha = {"68": 0.15, "90": 0.07}

fig, ax = plt.subplots(figsize=(11, 5))

for lam in lambdas:
    cb    = results[lam]["cb"]
    color = styles[lam]["color"]
    label = styles[lam]["label"]
    ax.fill_between(horizons, cb["lo90"], cb["hi90"],
                    alpha=fill_alpha["90"], color=color)
    ax.fill_between(horizons, cb["lo68"], cb["hi68"],
                    alpha=fill_alpha["68"], color=color)
    ax.plot(horizons, cb["med"], color=color, lw=2, label=label)

ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title(
    "Robustness: IRF — WIBOR 3M Shock $\\rightarrow$ CPI YoY\n"
    "BVAR(2) · Minnesota Prior · Shaded = 68% & 90% credible bands",
    fontsize=11,
)
ax.set_xlabel("Horizon (quarters)")
ax.set_ylabel("Response (CPI YoY, %)")
ax.set_xticks(horizons)
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "irf_robustness.png"), dpi=150)
plt.close(fig)
print("Saved: outputs/irf_robustness.png")

# ── 8. Results text ───────────────────────────────────────────────────────────
sep  = "=" * 70
dash = "-" * 65

reg_names = []
for lag in range(1, p + 1):
    for var in range(n):
        reg_names.append(f"{cols[var]}_L{lag}")
reg_names.append("intercept")

with open(os.path.join(OUTDIR, "step3_robustness.txt"), "w", encoding="utf-8") as f:
    f.write(f"{sep}\n  Step 3: Robustness — Minnesota Prior Sensitivity\n{sep}\n")
    f.write(f"Variables: {cols}\n")
    f.write(f"Lags p={p}, Gibbs draws kept={N_KEEP} (burn-in={N_BURN})\n")
    f.write(f"Models: lambda = {lambdas}\n\n")

    for lam in lambdas:
        B_mean = results[lam]["B_draws"].mean(0)
        B_std  = results[lam]["B_draws"].std(0)
        B_q16  = np.percentile(results[lam]["B_draws"], 16, 0)
        B_q84  = np.percentile(results[lam]["B_draws"], 84, 0)
        cb     = results[lam]["cb"]

        f.write(f"\n{sep}\n  lambda = {lam}\n{sep}\n")

        hdr = f"  {'Regressor':<22} {'Mean':>9} {'Std':>9} {'16%':>9} {'84%':>9}\n"
        for eq in range(n):
            f.write(f"\n  Equation: {cols[eq]}\n{hdr}")
            f.write("  " + "-" * 58 + "\n")
            for row, name in enumerate(reg_names):
                f.write(
                    f"  {name:<22} {B_mean[row,eq]:>9.4f} {B_std[row,eq]:>9.4f}"
                    f" {B_q16[row,eq]:>9.4f} {B_q84[row,eq]:>9.4f}\n"
                )

        f.write(f"\n  IRF: wibor_3m shock -> cpi_yoy  (lambda={lam})\n")
        f.write(f"  {'h':>3}  {'Median':>10}  {'16%':>10}  {'84%':>10}"
                f"  {'5%':>10}  {'95%':>10}\n")
        f.write("  " + "-" * 58 + "\n")
        for h in range(H + 1):
            f.write(
                f"  {h:>3}  {cb['med'][h]:>10.6f}  {cb['lo68'][h]:>10.6f}"
                f"  {cb['hi68'][h]:>10.6f}  {cb['lo90'][h]:>10.6f}"
                f"  {cb['hi90'][h]:>10.6f}\n"
            )

    # Comparison table: median IRF across lambdas
    f.write(f"\n{sep}\n  IRF Median Comparison (wibor_3m -> cpi_yoy)\n{sep}\n")
    f.write(f"  {'h':>3}" + "".join(f"  {'lam='+str(l):>12}" for l in lambdas) + "\n")
    f.write("  " + "-" * (3 + 14 * len(lambdas)) + "\n")
    for h in range(H + 1):
        row_str = f"  {h:>3}" + "".join(
            f"  {results[l]['cb']['med'][h]:>12.6f}" for l in lambdas
        )
        f.write(row_str + "\n")

print("Saved: outputs/step3_robustness.txt")
print("\nStep 3 complete.")
