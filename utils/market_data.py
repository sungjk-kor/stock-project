from __future__ import annotations

import pandas as pd
import yfinance as yf


MARKET_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "vix": "^VIX",
    "xli": "XLI",
    "sox": "^SOX",
}


def fetch_market_data(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch market index prices from Yahoo Finance."""
    frames: list[pd.Series] = []
    for name, ticker in MARKET_TICKERS.items():
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            continue
        close = _download_column(data, "Close").rename(name)
        frames.append(close)

        volume = _download_column(data, "Volume")
        if volume is not None and volume.notna().any():
            frames.append(volume.rename(f"{name}_volume"))

    if not frames:
        return pd.DataFrame()

    market = pd.concat(frames, axis=1)
    market.index.name = "date"
    return market.reset_index()


def _download_column(data: pd.DataFrame, column: str) -> pd.Series | None:
    """Return a yfinance column even when yfinance uses a MultiIndex."""
    if isinstance(data.columns, pd.MultiIndex):
        matches = [col for col in data.columns if col[0] == column]
        if not matches:
            return None
        return data[matches[0]]

    if column not in data.columns:
        return None
    return data[column]
