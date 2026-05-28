from __future__ import annotations

import os

import pandas as pd
from fredapi import Fred


RATE_SERIES = {
    "fed_funds": "FEDFUNDS",
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
}


def fetch_rates_data(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch interest-rate series from FRED."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames: list[pd.Series] = []
    for name, series_id in RATE_SERIES.items():
        series = fred.get_series(series_id, observation_start=start)
        frames.append(series.rename(name))

    rates = pd.concat(frames, axis=1)
    rates["term_spread_10y_2y"] = rates["treasury_10y"] - rates["treasury_2y"]
    rates.index.name = "date"
    return rates.reset_index()
