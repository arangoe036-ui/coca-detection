"""v2.3 Part B — official-census trend projection (the magnitude/trajectory layer).

Fits a trend to the OFFICIAL coca hectares (SIMCI via datos.gov.co, 2019–2024)
for the AOI and projects 2025–2026 with prediction intervals that widen with
extrapolation distance. This is the *trusted magnitude* layer — computed ONLY from
official data, never from the model's detected hectares (LOYO showed the model's
year-to-year magnitude is unreliable, 0.59–1.48, so using it for trend would
manufacture a fake trajectory).

**Framing (mandatory): a scenario — "if the 2019–2024 trajectory holds" — not a
forecast.** Coca is policy/market-driven and has reversed sharply before; the
intervals are wide and the projection must never be stated as fact.

    python -m src.trend

Writes ui/data/trend.json for the UI trajectory chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.data.labels import fetch_coca_grid
from src.data.multiyear import GRID_FIELD
from src.utils import load_config


def official_aoi_series(cfg, years):
    """Official coca hectares summed over the AOI bbox, per year."""
    out = {}
    for y in years:
        ycfg = {**cfg, "year": y, "labels": {**cfg["labels"], "coca_grid_year_field": GRID_FIELD[y]}}
        out[y] = float(fetch_coca_grid(ycfg, tuple(cfg["aoi"]["bbox"]))["coca_ha"].sum())
    return out


def loglinear_projection(years, ha, project_years, alpha=0.20):
    """Log-linear OLS (constant growth rate) with prediction intervals.

    Returns list of {year, mean, lo, hi} for project_years. alpha=0.20 → 80% PI.
    """
    x = np.asarray(years, float)
    y = np.log(np.asarray(ha, float))
    n = len(x)
    xbar = x.mean()
    Sxx = ((x - xbar) ** 2).sum()
    b, a = np.polyfit(x, y, 1)          # slope, intercept
    resid = y - (a + b * x)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    rows = []
    for x0 in project_years:
        yhat = a + b * x0
        se = s * np.sqrt(1 + 1 / n + (x0 - xbar) ** 2 / Sxx)  # prediction (widens with distance)
        rows.append({"year": int(x0), "mean": float(np.exp(yhat)),
                     "lo": float(np.exp(yhat - tcrit * se)), "hi": float(np.exp(yhat + tcrit * se))})
    growth = float(np.exp(b) - 1)       # annual multiplicative growth rate
    return rows, growth


def build_trend(cfg, hist_years=(2019, 2020, 2021, 2022, 2023, 2024), project=(2025, 2026)):
    series = official_aoi_series(cfg, list(hist_years))
    years = list(hist_years)
    ha = [series[y] for y in years]
    proj, growth = loglinear_projection(years, ha, list(project))

    out = {
        "aoi": cfg["aoi"]["region"],
        "model": "log-linear OLS on official AOI hectares",
        "annual_growth_rate": growth,
        "official": [{"year": y, "ha": series[y]} for y in years],
        "projection": proj,
        "framing": ("SCENARIO, not a forecast: 'if the 2019-2024 trajectory holds'. "
                    "Coca is policy/market-driven and has reversed before; intervals widen with "
                    "extrapolation. Magnitude/trend = official census; location = model. Never fused."),
    }
    ui_dir = Path(cfg["paths"]["ui_data_dir"]); ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "trend.json").write_text(json.dumps(out, indent=2))

    print(f"[trend] official AOI hectares {years[0]}–{years[-1]}:")
    for y in years:
        print(f"          {y}: {series[y]:,.0f} ha")
    print(f"[trend] annual growth rate ≈ {growth*100:+.1f}%/yr (log-linear)")
    for r in proj:
        print(f"[trend] projected {r['year']}: {r['mean']:,.0f} ha  (80% PI {r['lo']:,.0f}–{r['hi']:,.0f}) — SCENARIO")
    print(f"[trend] wrote {ui_dir/'trend.json'}")
    return out


if __name__ == "__main__":
    build_trend(load_config())
