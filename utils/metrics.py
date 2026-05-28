from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return float((true_direction == pred_direction).mean() * 100)
