from __future__ import annotations

import pandas as pd

from qplatform.features.indicators import simple_moving_average, relative_strength_index


class SmaCrossoverStrategy:
    def __init__(self, fast_window: int = 20, slow_window: int = 100) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be < slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_positions(self, close: pd.Series) -> pd.Series:
        fast = simple_moving_average(close, self.fast_window)
        slow = simple_moving_average(close, self.slow_window)
        raw_signal = (fast > slow).astype(int) - (fast < slow).astype(int)
        # Convert {-1,0,1} signals to positions; here we use {0,1}
        positions = raw_signal.clip(lower=0)  # long-only for simplicity
        positions.name = "position"
        return positions


class RsiMeanReversionStrategy:
    def __init__(self, window: int = 14, lower: int = 30, upper: int = 70) -> None:
        self.window = window
        self.lower = lower
        self.upper = upper

    def generate_positions(self, close: pd.Series) -> pd.Series:
        rsi = relative_strength_index(close, window=self.window)
        # Long when oversold, flat when overbought
        positions = ((rsi < self.lower).astype(int) * 1 + (rsi > self.upper).astype(int) * 0)
        positions.name = "position"
        return positions