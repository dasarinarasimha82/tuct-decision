from __future__ import annotations

import numpy as np
import pandas as pd


def annualization_factor(freq: str = "daily") -> int:
    if freq == "daily":
        return 252
    if freq == "weekly":
        return 52
    if freq == "monthly":
        return 12
    raise ValueError("Unsupported frequency")


def compute_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if equity_curve.empty:
        return float("nan")
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    num_periods = len(equity_curve)
    years = max(num_periods / periods_per_year, 1e-9)
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess.clip(upper=0)
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve / running_max) - 1.0
    return drawdowns.min()