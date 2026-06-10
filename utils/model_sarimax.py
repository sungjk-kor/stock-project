from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


def scale_exog(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exog_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    if not exog_cols:
        return train, test, None

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_test = test.copy()
    scaled_train[exog_cols] = scaler.fit_transform(train[exog_cols])
    scaled_test[exog_cols] = scaler.transform(test[exog_cols])
    return scaled_train, scaled_test, scaler


def find_auto_arima_order(
    train: pd.DataFrame,
    target_col: str,
    exog_cols: list[str] | None = None,
    seasonal: bool = False,
    m: int = 1,
):
    try:
        from pmdarima import auto_arima
    except ImportError as exc:
        raise ImportError("pmdarima is not installed. Run `pip install pmdarima`.") from exc

    exog = train[exog_cols] if exog_cols else None
    model = auto_arima(
        train[target_col],
        X=exog,
        seasonal=seasonal,
        m=m,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        max_d=2,
        max_P=2,
        max_Q=2,
        max_D=1,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    return model.order, model.seasonal_order, float(model.aic())


def fit_sarimax(
    train: pd.DataFrame,
    target_col: str = "sp500_log_return",
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
):
    exog = train[exog_cols] if exog_cols else None
    model = SARIMAX(
        train[target_col],
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
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


def rolling_forecast_sarimax(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    scale_external: bool = True,
) -> pd.Series:
    predictions: list[float] = []

    for step in range(len(test)):
        rolling_train = pd.concat([train, test.iloc[:step]], ignore_index=True)
        current_test = test.iloc[[step]].copy()

        if exog_cols and scale_external:
            rolling_train, current_test, _ = scale_exog(rolling_train, current_test, exog_cols)

        fitted = fit_sarimax(
            rolling_train,
            target_col=target_col,
            exog_cols=exog_cols,
            order=order,
            seasonal_order=seasonal_order,
        )
        future_exog = current_test[exog_cols] if exog_cols else None
        forecast = forecast_sarimax(fitted, steps=1, future_exog=future_exog)
        predictions.append(float(forecast.iloc[0]))

    return pd.Series(predictions, index=test.index, name="prediction")


def add_log_price_target(df: pd.DataFrame, price_col: str = "sp500", target_col: str = "sp500_log_price") -> pd.DataFrame:
    result = df.copy()
    result[target_col] = np.log(result[price_col])
    return result


def train_test_split_by_ratio(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_test_split_by_forecast_window(
    df: pd.DataFrame,
    forecast_start,
    forecast_end,
    train_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = df.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    forecast_start = pd.to_datetime(forecast_start)
    forecast_end = pd.to_datetime(forecast_end)

    test = dated[(dated["date"] >= forecast_start) & (dated["date"] <= forecast_end)].copy()
    train_candidates = dated[dated["date"] < forecast_start].copy()
    train = train_candidates.tail(train_weeks).copy()
    return train, test


def build_future_exog(df: pd.DataFrame, exog_cols: list[str], forecast_horizon: int) -> pd.DataFrame:
    """미래 예측용 외생변수 행렬 생성.

    lag 컬럼 {base}_lag_{k} 에 대해 예측 step i (1-indexed) 에서:
    - k >= i  → T-(k-i) 위치의 실제 관측값 사용 (알려진 값)
    - k < i   → 미래값이 필요하므로 base_col 의 마지막 관측값으로 대체
    lag 형식이 아닌 컬럼은 마지막 관측값 사용.
    """
    future_rows = []
    for step in range(1, forecast_horizon + 1):
        row: dict[str, float] = {}
        for col in exog_cols:
            lag_match = re.match(r"^(.+)_lag_(\d+)$", col)
            if lag_match:
                base_col = lag_match.group(1)
                k = int(lag_match.group(2))
                hist_steps_back = k - step  # >= 0 이면 알려진 과거값
                if hist_steps_back >= 0 and base_col in df.columns:
                    lookup = 1 + hist_steps_back
                    if lookup <= len(df):
                        val = float(df[base_col].iloc[-lookup])
                    else:
                        val = float(df[base_col].iloc[0])
                else:
                    if base_col in df.columns:
                        val = float(df[base_col].iloc[-1])
                    elif col in df.columns:
                        val = float(df[col].iloc[-1])
                    else:
                        val = 0.0
            else:
                val = float(df[col].iloc[-1]) if col in df.columns else 0.0
            row[col] = val
        future_rows.append(row)
    return pd.DataFrame(future_rows, columns=exog_cols)
