import sys, os
import pandas as pd
from datetime import datetime, timezone
sys.path.append('.')
from backtest.mock_market_service import MockMarketService
from backtest.fetch_datasets import PERIODS

p = PERIODS[0]
hist_file = f"backtest/data/historical_bars_{p['start']}_{p['end']}.csv"
daily_file = f"backtest/data/daily_bars_{p['start']}_{p['end']}.csv"
print(f"Starting MockMarketService init for {p['name']}")
ms = MockMarketService(current_time=pd.Timestamp.utcnow(), data_file=hist_file, daily_file=daily_file)
print("Done!")
