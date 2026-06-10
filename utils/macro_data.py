from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


MACRO_SERIES = {
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "unemployment": "UNRATE",
    "nonfarm_payroll": "PAYEMS",
    "retail_sales": "RSAFS",
    "industrial_production": "INDPRO",
}


def fetch_macro_data(start: str = "2000-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Fetch macroeconomic series from FRED."""
    load_dotenv()
    api_key = api_key or os.getenv("FRED_API_KEY")

    frames: list[pd.Series] = []
    if api_key:
        fred = Fred(api_key=api_key)
        for name, series_id in MACRO_SERIES.items():
            series = _get_series_safe(fred, series_id, start)
            if series is not None:
                frames.append(series.rename(name))

    local_ism_pmi = load_local_ism_pmi(start=start)
    if not local_ism_pmi.empty:
        frames.append(local_ism_pmi.set_index("date")["ism_pmi"])

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    return macro.reset_index()


def _get_series_safe(fred: Fred, series_id: str, start: str) -> pd.Series | None:
    try:
        return fred.get_series(series_id, observation_start=start)
    except ValueError:
        return None


def load_local_ism_pmi(start: str = "2000-01-01") -> pd.DataFrame:
    """Load ISM PMI from a local CSV because it has no stable yfinance ticker."""
    from utils.preprocessing import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "raw" / "ism_pmi.csv"
    if not path.exists():
        return pd.DataFrame()

    ism = pd.read_csv(path, parse_dates=["date"])
    if ism.empty or "ism_pmi" not in ism.columns:
        return pd.DataFrame()

    ism["ism_pmi"] = pd.to_numeric(ism["ism_pmi"], errors="coerce")
    ism = ism.dropna(subset=["date", "ism_pmi"])
    ism = ism[ism["date"] >= pd.to_datetime(start)]
    return ism[["date", "ism_pmi"]]
