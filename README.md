A minimal quantitative research scaffold with CLI, data ingestion (yfinance + synthetic fallback), features, baseline models, backtesting, and risk metrics.

Quickstart:

1) Install deps:
   pip install -r requirements.txt

2) Run a quick synthetic sanity backtest:
   python scripts/run_sanity_backtest.py

3) Run on live symbol (falls back to synthetic if download fails):
   python -m qplatform.cli backtest --symbol SPY --start 2018-01-01 --end 2024-01-01 --strategy sma --fast 20 --slow 100 --tc_bps 1.0