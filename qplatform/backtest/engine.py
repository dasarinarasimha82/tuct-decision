from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from qplatform.risk.metrics import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    returns: pd.Series
    strategy_returns: pd.Series
    metrics: Dict[str, float]


def run_backtest(close: pd.Series, positions: pd.Series, transaction_cost_bps: float = 1.0) -> BacktestResult:
    close = close.dropna()
    positions = positions.reindex(close.index).fillna(method="ffill").fillna(0.0)

    returns = close.pct_change().fillna(0.0)

    # Strategy returns are based on prior-day position
    strat_returns_before_costs = positions.shift(1).fillna(0.0) * returns

    # Transaction costs proportional to turnover
    turnover = positions.diff().abs().fillna(0.0)
    cost_rate = transaction_cost_bps / 10_000.0
    costs = turnover * cost_rate

    strategy_returns = strat_returns_before_costs - costs

    equity_curve = (1.0 + strategy_returns).cumprod()

    metrics = {
        "CAGR": compute_cagr(equity_curve, periods_per_year=252),
        "Sharpe": compute_sharpe(strategy_returns, periods_per_year=252),
        "Sortino": compute_sortino(strategy_returns, periods_per_year=252),
        "MaxDrawdown": compute_max_drawdown(equity_curve),
        "TotalReturn": equity_curve.iloc[-1] - 1.0,
    }

    return BacktestResult(
        equity_curve=equity_curve,
        positions=positions,
        returns=returns,
        strategy_returns=strategy_returns,
        metrics=metrics,
    )