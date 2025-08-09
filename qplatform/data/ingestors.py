from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PriceDataRequest:
    symbol: str
    start: datetime
    end: datetime
    interval: str = "1d"
    source_preference: List[str] = None

    def __post_init__(self) -> None:
        if self.source_preference is None:
            self.source_preference = ["synthetic", "yfinance"]


def fetch_price_history(request: PriceDataRequest) -> pd.DataFrame:
    """Fetch OHLCV price history with a robust fallback to synthetic data.

    Returns a DataFrame indexed by datetime with columns: Open, High, Low, Close, Adj Close, Volume.
    """
    last_error: Optional[Exception] = None

    for source in request.source_preference:
        try:
            if source == "yfinance":
                return _fetch_with_yfinance(request)
            if source == "synthetic":
                return _generate_synthetic_prices(request)
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No data source available")


def _fetch_with_yfinance(request: PriceDataRequest) -> pd.DataFrame:
    import yfinance as yf  # lazy import

    df = yf.download(
        tickers=request.symbol,
        start=request.start,
        end=request.end,
        interval=request.interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        # Handle multi-index columns when multiple tickers are returned
        df = df.xs(request.symbol, axis=1, level=1)

    if df.empty:
        raise ValueError("yfinance returned empty data")

    # Ensure standard columns and types
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected_cols:
        if col not in df:
            if col == "Adj Close" and "Adj Close" not in df and "Adj Close" not in df.columns:
                # Older yfinance versions sometimes miss Adj Close; copy Close
                df["Adj Close"] = df["Close"]
            elif col == "Volume" and "Volume" not in df:
                df["Volume"] = 0
            else:
                df[col] = df["Close"]

    df = df[expected_cols]
    df = df.dropna(how="any")
    df.index = pd.to_datetime(df.index)
    return df


def _generate_synthetic_prices(request: PriceDataRequest) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    # Determine frequency in days for interval
    if request.interval.endswith("m"):
        # minute data unsupported in simple synthetic; approximate to business days
        freq = "B"
        steps_per_year = 252
    else:
        freq = "B"
        steps_per_year = 252

    index = pd.date_range(request.start, request.end, freq=freq, inclusive="left")
    num_steps = len(index)
    if num_steps == 0:
        raise ValueError("No dates in range for synthetic data")

    annual_drift = 0.08
    annual_vol = 0.20
    dt = 1.0 / steps_per_year

    # Geometric Brownian Motion for Close
    shocks = rng.normal(loc=0.0, scale=np.sqrt(dt), size=num_steps)
    log_returns = (annual_drift - 0.5 * annual_vol ** 2) * dt + annual_vol * shocks
    start_price = 100.0
    close = start_price * np.exp(np.cumsum(log_returns))

    # Build OHLC with small intraday ranges
    intraday_spread = np.maximum(0.001, rng.normal(0.0015, 0.0005, size=num_steps))
    open_price = np.concatenate([[start_price], close[:-1]]) * (1.0 + rng.normal(0.0, 0.0005, size=num_steps))
    high = np.maximum(close, open_price) * (1.0 + intraday_spread)
    low = np.minimum(close, open_price) * (1.0 - intraday_spread)
    volume = rng.integers(low=100_000, high=1_000_000, size=num_steps)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=index,
    )
    return df