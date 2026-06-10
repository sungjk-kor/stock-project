from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


RATE_SERIES = {
    "fed_funds": "FEDFUNDS",
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
}


def fetch_rates_data(start: str = "2000-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Fetch interest-rate series from FRED."""
    load_dotenv()
    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames: list[pd.Series] = []
    for name, series_id in RATE_SERIES.items():
        series = _get_series_safe(fred, series_id, start)
        if series is not None:
            frames.append(series.rename(name))

    if not frames:
        return pd.DataFrame()

    rates = pd.concat(frames, axis=1)
    if {"treasury_10y", "treasury_2y"}.issubset(rates.columns):
        rates["term_spread_10y_2y"] = rates["treasury_10y"] - rates["treasury_2y"]
    rates.index.name = "date"
    return rates.reset_index()


def _get_series_safe(fred: Fred, series_id: str, start: str) -> pd.Series | None:
    try:
        return fred.get_series(series_id, observation_start=start)
    except ValueError:
        return None
