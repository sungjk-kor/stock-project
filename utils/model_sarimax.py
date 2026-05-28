from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarimax(
    train: pd.DataFrame,
    target_col: str = "sp500_log_return",
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
):
    exog = train[exog_cols] if exog_cols else None
    model = SARIMAX(
        train[target_col],
        exog=exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax(
    fitted_model,
    steps: int,
    future_exog: pd.DataFrame | None = None,
) -> pd.Series:
    forecast = fitted_model.get_forecast(steps=steps, exog=future_exog)
    return forecast.predicted_mean


def train_test_split_by_ratio(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
