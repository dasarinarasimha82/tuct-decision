from __future__ import annotations

import numpy as np
import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    if span <= 0:
        raise ValueError("span must be positive")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi