from __future__ import annotations

import os

import pandas as pd
from fredapi import Fred


MACRO_SERIES = {
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "unemployment": "UNRATE",
    "nonfarm_payroll": "PAYEMS",
    "retail_sales": "RSAFS",
    "industrial_production": "INDPRO",
}


def fetch_macro_data(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch macroeconomic series from FRED."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames = [
        fred.get_series(series_id, observation_start=start).rename(name)
        for name, series_id in MACRO_SERIES.items()
    ]
    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    return macro.reset_index()
