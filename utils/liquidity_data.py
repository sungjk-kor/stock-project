from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import yfinance as yf


FRED_LIQUIDITY_SERIES = {
    "m2": "M2SL",
    "financial_conditions": "NFCI",
    # Chicago Fed NFCI 레버리지 서브인덱스 (주간, 이미 표준화)
    # 양수 = 레버리지 긴축/위험회피, 음수 = 레버리지 확대/위험선호
    "nfci_leverage": "NFCILEVERAGE",
    # Moody's Baa 회사채 - 10년 국채 신용스프레드 (일별)
    # ICE BofA HY OAS(BAMLH0A0HYM2)는 FRED API 접근 제한으로 대체
    "credit_spread": "BAA10Y",
}

YAHOO_LIQUIDITY_TICKERS = {
    "dxy": "DX-Y.NYB",
    "oil": "CL=F",
    "gold": "GC=F",
}


def fetch_liquidity_data(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch liquidity, dollar, and commodity indicators."""
    frames: list[pd.Series] = []

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        fred = Fred(api_key=api_key)
        for name, series_id in FRED_LIQUIDITY_SERIES.items():
            series = _get_series_safe(fred, series_id, start)
            if series is not None:
                frames.append(series.rename(name))

    for name, ticker in YAHOO_LIQUIDITY_TICKERS.items():
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not data.empty:
            frames.append(_download_column(data, "Close").rename(name))

    if not frames:
        return pd.DataFrame()

    liquidity = pd.concat(frames, axis=1)
    liquidity.index.name = "date"
    return liquidity.reset_index()


def _download_column(data: pd.DataFrame, column: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        matches = [col for col in data.columns if col[0] == column]
        return data[matches[0]]
    return data[column]


def _get_series_safe(fred: Fred, series_id: str, start: str) -> pd.Series | None:
    try:
        return fred.get_series(series_id, observation_start=start)
    except ValueError:
        return None
