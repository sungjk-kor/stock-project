from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from utils.preprocessing import PROJECT_ROOT


SECTOR_ETF_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sector_etf_weekly.csv"

SECTOR_ETFS: dict[str, str] = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

# 그룹2: 테마·전략형 ETF (성장, 기술, AI, 생명공학, 변동성 등)
SECTOR_ETFS_GROUP2: dict[str, str] = {
    "CIBR": "Cybersecurity",
    "SOXX": "Semiconductors",
    "AIQ": "AI & Machine Learning",
    "BOTZ": "Robotics & AI",
    "VGT": "Technology",
    "ARKK": "Innovation",
    "XBI": "Biotechnology",
    "ARKX": "Space Exploration",
    "ARKG": "Genomics",
    "MCHI": "China Tech",
    "TQQQ": "NASDAQ 3x Leverage",
    "SQQQ": "NASDAQ 3x Inverse",
    "PSQ": "QQQ 3x Inverse",
    "TMF": "20Y Bond 3x Leverage",
    "VIXY": "VIX 4x Leverage",
    "USO": "Oil Prices",
}

EVENT_LABELS = {
    "golden_cross": "주가 골드크로스",
    "dead_cross": "주가 데드크로스",
    "volume_spike": "거래량 급증",
    "low_rebound_20": "저점대비 20% 상승",
    "high_drawdown_20": "고점대비 20% 하락",
}


def fetch_sector_etf_weekly(
    start: str = "2020-01-01",
    end: str | None = None,
    tickers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch weekly close and volume for liquid US sector ETFs."""
    ticker_map = tickers or SECTOR_ETFS
    frames: list[pd.DataFrame] = []

    for ticker, sector in ticker_map.items():
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            continue

        close = _download_column(raw, "Close")
        volume = _download_column(raw, "Volume")
        if close is None or volume is None:
            continue

        daily = pd.DataFrame({"close": close, "volume": volume})
        daily.index = pd.to_datetime(daily.index)
        weekly = pd.DataFrame(
            {
                "close": daily["close"].resample("W-FRI").last(),
                "volume": daily["volume"].resample("W-FRI").sum(),
            }
        ).dropna(subset=["close"])

        weekly["ticker"] = ticker
        weekly["sector"] = sector
        weekly.index.name = "date"
        frames.append(weekly.reset_index())

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "sector", "close", "volume"])

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    result["weekly_return"] = result.groupby("ticker")["close"].pct_change()
    return result[["date", "ticker", "sector", "close", "volume", "weekly_return"]]


def save_sector_etf_weekly(df: pd.DataFrame, path: str | Path = SECTOR_ETF_DATA_PATH) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_sector_etf_weekly(path: str | Path = SECTOR_ETF_DATA_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(input_path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def collect_sector_etf_weekly(
    start: str = "2020-01-01",
    end: str | None = None,
    path: str | Path = SECTOR_ETF_DATA_PATH,
) -> pd.DataFrame:
    # 그룹1과 그룹2의 ETF를 모두 수집
    all_etfs = {**SECTOR_ETFS, **SECTOR_ETFS_GROUP2}
    df = fetch_sector_etf_weekly(start=start, end=end, tickers=all_etfs)
    if not df.empty:
        save_sector_etf_weekly(df, path=path)
    return df


def add_sector_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving-average and threshold fields used by the event monitor."""
    if df.empty:
        return df

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = result.groupby("ticker", group_keys=False)
    result["ma_12w"] = grouped["close"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    result["ma_26w"] = grouped["close"].transform(lambda s: s.rolling(26, min_periods=26).mean())
    result["volume_ma_12w"] = grouped["volume"].transform(lambda s: s.rolling(12, min_periods=6).mean())
    result["rolling_low_52w"] = grouped["close"].transform(lambda s: s.rolling(52, min_periods=26).min())
    result["rolling_high_52w"] = grouped["close"].transform(lambda s: s.rolling(52, min_periods=26).max())
    return result


def detect_sector_etf_events(
    df: pd.DataFrame,
    volume_spike_multiple: float = 2.0,
) -> pd.DataFrame:
    """Detect notable weekly price and volume events for sector ETFs."""
    featured = add_sector_event_features(df)
    if featured.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ticker, part in featured.groupby("ticker"):
        part = part.sort_values("date").reset_index(drop=True)
        sector = str(part["sector"].iloc[-1]) if "sector" in part else ticker

        prev_ma_diff = (part["ma_12w"] - part["ma_26w"]).shift(1)
        ma_diff = part["ma_12w"] - part["ma_26w"]
        golden = (prev_ma_diff <= 0) & (ma_diff > 0)
        dead = (prev_ma_diff >= 0) & (ma_diff < 0)

        volume_spike = (
            part["volume_ma_12w"].notna()
            & (part["volume"] >= part["volume_ma_12w"] * volume_spike_multiple)
        )

        rebound_level = part["rolling_low_52w"] * 1.2
        drawdown_level = part["rolling_high_52w"] * 0.8
        low_rebound = (part["close"].shift(1) < rebound_level.shift(1)) & (part["close"] >= rebound_level)
        high_drawdown = (part["close"].shift(1) > drawdown_level.shift(1)) & (part["close"] <= drawdown_level)

        event_masks = {
            "golden_cross": golden,
            "dead_cross": dead,
            "volume_spike": volume_spike,
            "low_rebound_20": low_rebound,
            "high_drawdown_20": high_drawdown,
        }

        for event_type, mask in event_masks.items():
            event_rows = part[mask.fillna(False)]
            for _, row in event_rows.iterrows():
                rows.append(_build_event_row(row, ticker, sector, event_type, volume_spike_multiple))

    if not rows:
        return pd.DataFrame(
            columns=["date", "ticker", "sector", "event_type", "event", "comment", "close", "volume"]
        )

    events = pd.DataFrame(rows)
    return events.sort_values(["date", "ticker", "event_type"], ascending=[False, True, True]).reset_index(drop=True)


def _build_event_row(
    row: pd.Series,
    ticker: str,
    sector: str,
    event_type: str,
    volume_spike_multiple: float,
) -> dict:
    close = float(row["close"])
    volume = float(row["volume"])
    event = EVENT_LABELS[event_type]

    if event_type == "golden_cross":
        detail = f"12주 이동평균이 26주 이동평균을 상향 돌파했습니다."
    elif event_type == "dead_cross":
        detail = f"12주 이동평균이 26주 이동평균을 하향 돌파했습니다."
    elif event_type == "volume_spike":
        avg_volume = float(row["volume_ma_12w"])
        multiple = volume / avg_volume if avg_volume else np.nan
        detail = f"주간 거래량이 12주 평균의 {multiple:.1f}배로 증가했습니다."
    elif event_type == "low_rebound_20":
        low = float(row["rolling_low_52w"])
        detail = f"52주 저점({low:,.2f}) 대비 20% 상승선을 돌파했습니다."
    elif event_type == "high_drawdown_20":
        high = float(row["rolling_high_52w"])
        detail = f"52주 고점({high:,.2f}) 대비 20% 하락선을 이탈했습니다."
    else:
        detail = "이벤트가 감지되었습니다."

    return {
        "date": row["date"],
        "ticker": ticker,
        "sector": sector,
        "event_type": event_type,
        "event": event,
        "comment": f"{ticker} ({sector}) - {event}: {detail}",
        "close": close,
        "volume": volume,
        "weekly_return": row.get("weekly_return", np.nan),
        "volume_spike_multiple": volume_spike_multiple,
    }


def _download_column(data: pd.DataFrame, column: str) -> pd.Series | None:
    if isinstance(data.columns, pd.MultiIndex):
        matches = [col for col in data.columns if col[0] == column]
        if not matches:
            return None
        return data[matches[0]]

    if column not in data.columns:
        return None
    return data[column]
