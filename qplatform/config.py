from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestConfig:
    symbol: str
    start_date: datetime
    end_date: datetime
    fast_window: int = 20
    slow_window: int = 100
    transaction_cost_bps: float = 1.0
    interval: str = "1d"