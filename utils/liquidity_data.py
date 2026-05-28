from __future__ import annotations

import os

import pandas as pd
from fredapi import Fred
import yfinance as yf


FRED_LIQUIDITY_SERIES = {
    "m2": "M2SL",
    "financial_conditions": "NFCI",
}

YAHOO_LIQUIDITY_TICKERS = {
    "dxy": "DX-Y.NYB",
    "oil": "CL=F",
    "gold": "GC=F",
}


def fetch_liquidity_data(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch liquidity, dollar, and commodity indicators."""
    frames: list[pd.Series] = []

    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        fred = Fred(api_key=api_key)
        for name, series_id in FRED_LIQUIDITY_SERIES.items():
            frames.append(fred.get_series(series_id, observation_start=start).rename(name))

    for name, ticker in YAHOO_LIQUIDITY_TICKERS.items():
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not data.empty:
            frames.append(data["Close"].rename(name))

    if not frames:
        return pd.DataFrame()

    liquidity = pd.concat(frames, axis=1)
    liquidity.index.name = "date"
    return liquidity.reset_index()
