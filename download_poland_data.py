"""
Download Polish macro data for BVAR analysis
Sources:
  - GDP YoY quarterly growth (%): Eurostat namq_10_gdp (CLV_PCH_SM, SCA, Q)
  - CPI YoY monthly->quarterly (%): Eurostat prc_hicp_manr (RCH_A, M)
  - WIBOR 3M quarterly average (%): Eurostat irt_st_m (IRT_M3, M) -> avg to Q
Sample: 2000Q1 - 2023Q4
Output: bvar_poland/data/poland_macro_data.csv
"""

import os
import requests
import pandas as pd

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "data")
os.makedirs(OUTDIR, exist_ok=True)

HEADERS   = {"Accept": "application/json"}
START_Q, END_Q = "2000-Q1", "2023-Q4"
START_M, END_M = "2000-01", "2023-12"


def _eurostat_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_quarterly(url: str, col: str) -> pd.Series:
    """Parse Eurostat JSON-stat (quarterly) into a pd.Series indexed by 'YYYY-QN'."""
    d = _eurostat_json(url)
    dims     = d["dimension"]
    time_idx = dims["time"]["category"]["index"]
    vals     = d["value"]
    rows = {
        period: vals[str(i)]
        for period, i in time_idx.items()
        if START_Q <= period <= END_Q and str(i) in vals
    }
    s = pd.Series(rows, name=col)
    s.index.name = "quarter"
    return s.sort_index()


def fetch_monthly_to_quarterly(url: str, col: str) -> pd.Series:
    """Download monthly Eurostat data, average to quarterly, return 'YYYY-QN' index."""
    d = _eurostat_json(url)
    dims     = d["dimension"]
    time_idx = dims["time"]["category"]["index"]
    vals     = d["value"]
    rows = {
        period: vals[str(i)]
        for period, i in time_idx.items()
        if START_M <= period <= END_M and str(i) in vals
    }
    monthly  = pd.Series(rows, name=col)
    monthly.index = pd.to_datetime(monthly.index)          # convert to DatetimeIndex
    quarterly = monthly.resample("QE").mean()               # quarter-end resample
    quarterly.index = quarterly.index.to_period("Q").strftime("%Y-Q%q")
    quarterly.index.name = "quarter"
    return quarterly.sort_index()


# ── 1. GDP YoY quarterly growth (%) ─────────────────────────────────────────
print("1. Downloading GDP YoY quarterly growth from Eurostat …")
gdp_url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "namq_10_gdp?geo=PL&na_item=B1GQ&unit=CLV_PCH_SM&s_adj=SCA&freq=Q"
)
gdp = fetch_quarterly(gdp_url, "gdp_yoy")
print(f"   GDP: {len(gdp)} quarters  ({gdp.index[0]} to {gdp.index[-1]})")

# ── 2. CPI YoY monthly → quarterly average (%) ──────────────────────────────
print("2. Downloading HICP CPI YoY monthly from Eurostat, averaging to quarterly …")
cpi_url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hicp_manr?geo=PL&coicop=CP00&unit=RCH_A&freq=M"
)
cpi = fetch_monthly_to_quarterly(cpi_url, "cpi_yoy")
print(f"   CPI: {len(cpi)} quarters  ({cpi.index[0]} to {cpi.index[-1]})")

# ── 3. WIBOR 3M monthly → quarterly average (%) ─────────────────────────────
print("3. Downloading WIBOR 3M monthly from Eurostat (irt_st_m), averaging to quarterly …")
wibor_url = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "irt_st_m?geo=PL&int_rt=IRT_M3&freq=M"
)
wibor = fetch_monthly_to_quarterly(wibor_url, "wibor_3m")
print(f"   WIBOR 3M: {len(wibor)} quarters  ({wibor.index[0]} to {wibor.index[-1]})")

# ── 4. Merge and align to 2000Q1–2023Q4 ─────────────────────────────────────
all_quarters = [f"{y}-Q{q}" for y in range(2000, 2024) for q in range(1, 5)]

df = (
    pd.DataFrame({"gdp_yoy": gdp, "cpi_yoy": cpi, "wibor_3m": wibor})
    .reindex(all_quarters)
    .sort_index()
)
df.index.name = "quarter"

print(f"\nMerged dataset shape : {df.shape}")
print("Missing values       :\n", df.isnull().sum())
print("\nFirst 8 rows:\n", df.head(8).to_string())
print("\nLast 4 rows:\n", df.tail(4).to_string())

# ── 5. Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTDIR, "poland_macro_data.csv")
df.to_csv(out_path)
print(f"\nSaved → {out_path}")
