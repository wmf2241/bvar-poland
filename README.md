# BVAR Analysis for Poland: Monetary Policy Transmission

A Bayesian Vector Autoregression (BVAR) study of monetary policy transmission in Poland, estimating how interest rate shocks propagate to output and inflation over 2000–2023.

## Overview

This project implements a full BVAR workflow in three steps:

1. **Data download & diagnostics** — fetch quarterly macro data from Eurostat, run unit-root and cointegration tests, select VAR lag order
2. **BVAR estimation** — Gibbs sampler with Minnesota prior, compute Impulse Response Functions (IRFs) with credible bands
3. **Robustness checks** — compare IRF results across tight, baseline, and loose prior specifications

**Sample period:** 2000-Q1 to 2023-Q4 (96 quarters)

**Variables:**
| Variable | Description | Source |
|---|---|---|
| `gdp_yoy` | GDP year-on-year growth rate (%) | Eurostat `namq_10_gdp` |
| `cpi_yoy` | HICP CPI year-on-year growth rate (%) | Eurostat `prc_hicp_manr` |
| `wibor_3m` | WIBOR 3M interest rate, quarterly average (%) | Eurostat `irt_st_m` |

## Repository Structure

```
bvar_poland/
├── download_poland_data.py     # Step 0: fetch data from Eurostat API
├── step1_bvar_poland.py        # Step 1: diagnostics (ADF, Johansen, lag selection)
├── step2_bvar_estimation.py    # Step 2: BVAR estimation & IRFs
├── step3_robustness.py         # Step 3: prior sensitivity robustness check
├── data/
│   └── poland_macro_data.csv   # Cleaned quarterly dataset (96 × 3)
├── outputs/
│   ├── poland_data_plot.png    # Time-series plot of all three variables
│   ├── poland_step1_results.txt
│   ├── irf_rate_gdp.png        # IRF: WIBOR shock → GDP YoY
│   ├── irf_rate_cpi.png        # IRF: WIBOR shock → CPI YoY
│   ├── diagnostics.png         # MCMC trace plots (2 chains, 8 parameters)
│   ├── step2_results.txt
│   ├── irf_robustness.png      # IRF comparison across λ = 0.1, 0.2, 0.4
│   └── step3_robustness.txt
├── poland_bvar_report.docx     # Full analysis report
└── bvar_poland_report final.pdf
```

## Methodology

### Model

A BVAR(2) model estimated in levels:

```
Y_t = B_1 Y_{t-1} + B_2 Y_{t-2} + c + ε_t,   ε_t ~ N(0, Σ)
```

where `Y_t = [gdp_yoy, cpi_yoy, wibor_3m]'`.

### Minnesota Prior

A standard Minnesota prior is placed on the VAR coefficients, with the random-walk assumption for own first lags:

- **Tightness parameter λ** controls the overall prior shrinkage
- Baseline: **λ = 0.2**
- Own first-lag prior mean = 1 (random walk); all other coefficients = 0
- Intercepts: diffuse prior (variance = 10⁶)
- Σ: inverse-Wishart prior with scale = diag(σ²_OLS)

### Estimation

- **Gibbs sampler:** block draws from the conditional posteriors of `vec(B) | Σ, Y` and `Σ | B, Y`
- **Chains:** 2 independent chains × 6,000 draws each (1,000 burn-in discarded)
- **Posterior draws kept:** 2 × 5,000 = 10,000
- **Convergence:** Gelman-Rubin R-hat diagnostic (all parameters < 1.01)

### Impulse Response Functions

Structural shocks are identified via Cholesky decomposition of the posterior residual covariance Σ. The ordering is `[gdp_yoy, cpi_yoy, wibor_3m]`, so the WIBOR shock is ordered last (i.e., it reacts contemporaneously to output and inflation). IRFs are computed for H = 20 quarters with 68% and 90% posterior credible bands.

## Key Results

### Step 1 — Diagnostics

| Variable | ADF (level) | Stationary? | ADF (Δ) | Stationary? |
|---|---|---|---|---|
| `gdp_yoy` | −4.50 (p=0.0002) | Yes | −6.62 | Yes |
| `cpi_yoy` | −1.87 (p=0.3464) | No | −5.69 | Yes |
| `wibor_3m` | −4.05 (p=0.0012) | Yes | −4.04 | Yes |

Johansen trace test finds **3 cointegrating vectors** at the 5% level; the model is estimated in levels to preserve cointegrating relationships. BIC selects **p = 2 lags**.

### Step 2 — Convergence

All Gelman-Rubin R-hat statistics are within [0.9999, 1.0003], confirming excellent MCMC convergence across both chains.

### Step 2 — IRF Findings

**WIBOR shock → GDP YoY:** The response turns negative by horizon 2–5 (peak around h=4, median ≈ −0.064%), then gradually reverts and turns slightly positive after horizon 10. The 68% credible band straddles zero at most horizons, indicating moderate uncertainty.

**WIBOR shock → CPI YoY:** A clearer deflationary transmission. The response turns negative from horizon 4 onward, peaking at around h=11 (median ≈ −0.196%). The 68% credible band is entirely below zero from h=7 onward, providing strong posterior evidence of a negative effect.

### Step 3 — Robustness

Prior sensitivity across λ ∈ {0.1, 0.2, 0.4} shows that the qualitative shape of the CPI IRF is stable: all three specifications produce a hump-shaped negative response peaking around horizon 10–12. A tighter prior (λ=0.1) yields a slightly muted response; a looser prior (λ=0.4) allows a slightly larger response amplitude.

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scipy
statsmodels
requests
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib scipy statsmodels requests
```

## Usage

Run the scripts in order:

```bash
# 1. Download data from Eurostat
python download_poland_data.py

# 2. Diagnostics
python step1_bvar_poland.py

# 3. BVAR estimation
python step2_bvar_estimation.py

# 4. Robustness checks
python step3_robustness.py
```

All outputs (plots and text files) are saved to the `outputs/` directory.

> **Note:** The pre-downloaded dataset `data/poland_macro_data.csv` is included in the repository, so you can skip `download_poland_data.py` and run steps 1–3 directly.

## Data Sources

- **Eurostat `namq_10_gdp`** — Quarterly national accounts; GDP chain-linked volumes, percentage change on same quarter of previous year, seasonally and calendar adjusted (SCA)
- **Eurostat `prc_hicp_manr`** — HICP monthly rates of change, all-items (CP00), averaged to quarterly
- **Eurostat `irt_st_m`** — Short-term interest rates, WIBOR 3M monthly, averaged to quarterly
