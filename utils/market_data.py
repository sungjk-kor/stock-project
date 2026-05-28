from __future__ import annotations

import pandas as pd
import yfinance as yf


MARKET_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "vix": "^VIX",
}


def fetch_market_data(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch market index prices from Yahoo Finance."""
    frames: list[pd.Series] = []
    for name, ticker in MARKET_TICKERS.items():
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            continue
        frames.append(data["Close"].rename(name))

    if not frames:
        return pd.DataFrame()

    market = pd.concat(frames, axis=1)
    market.index.name = "date"
    return market.reset_index()
