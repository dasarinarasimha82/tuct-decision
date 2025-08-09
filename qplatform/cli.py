from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict

import pandas as pd

from qplatform.config import BacktestConfig
from qplatform.data.ingestors import PriceDataRequest, fetch_price_history
from qplatform.models.baselines import RsiMeanReversionStrategy, SmaCrossoverStrategy
from qplatform.backtest.engine import run_backtest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qplatform", description="Quant Research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanity = subparsers.add_parser("sanity", help="Run a quick synthetic backtest")
    sanity.add_argument("--years", type=int, default=5)

    backtest = subparsers.add_parser("backtest", help="Run a backtest on a symbol")
    backtest.add_argument("--symbol", type=str, default="SPY")
    backtest.add_argument("--start", type=str, default="2018-01-01")
    backtest.add_argument("--end", type=str, default="2024-01-01")
    backtest.add_argument("--interval", type=str, default="1d")
    backtest.add_argument("--strategy", type=str, choices=["sma", "rsi"], default="sma")
    backtest.add_argument("--fast", type=int, default=20)
    backtest.add_argument("--slow", type=int, default=100)
    backtest.add_argument("--rsi_window", type=int, default=14)
    backtest.add_argument("--rsi_lower", type=int, default=30)
    backtest.add_argument("--rsi_upper", type=int, default=70)
    backtest.add_argument("--tc_bps", type=float, default=1.0, help="Transaction costs in bps per unit turnover")

    return parser.parse_args()


def _run_sanity(years: int = 5) -> Dict[str, float]:
    end = datetime.utcnow()
    start = datetime(end.year - years, end.month, end.day)

    req = PriceDataRequest(symbol="SYNTH", start=start, end=end, interval="1d", source_preference=["synthetic"]) 
    df = fetch_price_history(req)

    strategy = SmaCrossoverStrategy(fast_window=20, slow_window=100)
    positions = strategy.generate_positions(df["Close"])  
    result = run_backtest(df["Close"], positions, transaction_cost_bps=1.0)

    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"EquityEnd: {result.equity_curve.iloc[-1]:.4f}")

    return result.metrics


def _run_backtest(args: argparse.Namespace) -> Dict[str, float]:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    config = BacktestConfig(
        symbol=args.symbol,
        start_date=start,
        end_date=end,
        fast_window=args.fast,
        slow_window=args.slow,
        transaction_cost_bps=args.tc_bps,
        interval=args.interval,
    )

    req = PriceDataRequest(symbol=config.symbol, start=config.start_date, end=config.end_date, interval=config.interval)
    df = fetch_price_history(req)

    if args.strategy == "sma":
        strategy = SmaCrossoverStrategy(fast_window=config.fast_window, slow_window=config.slow_window)
        positions = strategy.generate_positions(df["Close"])  
    else:
        strategy = RsiMeanReversionStrategy(window=args.rsi_window, lower=args.rsi_lower, upper=args.rsi_upper)
        positions = strategy.generate_positions(df["Close"])  

    result = run_backtest(df["Close"], positions, transaction_cost_bps=config.transaction_cost_bps)

    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"EquityEnd: {result.equity_curve.iloc[-1]:.4f}")

    return result.metrics


def main() -> None:
    args = _parse_args()
    if args.command == "sanity":
        _run_sanity(years=args.years)
    elif args.command == "backtest":
        _run_backtest(args)


if __name__ == "__main__":
    main()