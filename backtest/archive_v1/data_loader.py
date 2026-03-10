import os
import sys
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# File paths are built dynamically in load_data now

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "SMCI", "PLTR",
    "MSTR", "COIN", "MARA",
    "RKLB", "IONQ", "ASTS", "SOUN",
    "LMT", "WMT", "JNJ", "XOM",
    "SPY", "QQQ", "IWM", "GLD"
]

def load_data(days_back=60, start_str=None, end_str=None):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    
    if start_str and end_str:
        end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_date_intraday = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # We still need 150 days prior to the start date for daily bars to build EMA100
        start_date_daily = start_date_intraday - timedelta(days=150)
        
        DATA_FILE = os.path.join(os.path.dirname(__file__), "data", f"historical_bars_{start_str}_{end_str}.csv")
        DAILY_FILE = os.path.join(os.path.dirname(__file__), "data", f"daily_bars_{start_str}_{end_str}.csv")
    else:
        # Offset by 20 minutes to avoid Alpaca Free Tier "recent SIP data" errors
        end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
        start_date_intraday = end_date - timedelta(days=days_back)
        start_date_daily = end_date - timedelta(days=150)
        
        DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "historical_bars.csv")
        DAILY_FILE = os.path.join(os.path.dirname(__file__), "data", "daily_bars.csv")

    print(f"Fetching intraday 5-min bars for {len(TICKERS)} tickers from {start_date_intraday.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Intraday 5-min bars
    try:
        tf_5m = TimeFrame(amount=5, unit=TimeFrameUnit.Minute)
        request_intraday = StockBarsRequest(
            symbol_or_symbols=TICKERS,
            timeframe=tf_5m,
            start=start_date_intraday,
            end=end_date,
            feed="iex"
        )
        bars_intraday = client.get_stock_bars(request_intraday).df
        if not bars_intraday.empty:
            # Drop timezone if needed and reset index
            bars_intraday.reset_index(inplace=True)
            bars_intraday.to_csv(DATA_FILE, index=False)
            print(f"Successfully saved {len(bars_intraday)} 5-min rows to {os.path.basename(DATA_FILE)}")
        else:
            print("Received empty Intraday DataFrame from Alpaca.")
    except Exception as e:
        safe_msg = repr(e).encode("ascii", "replace").decode("ascii")
        print(f"Error fetching 5-min bars: {safe_msg}")

    print(f"Fetching daily bars for {len(TICKERS)} tickers from {start_date_daily.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Daily bars (for Trend Baseline like EMA100, RSI, MACD, etc.)
    try:
        tf_day = TimeFrame.Day
        request_daily = StockBarsRequest(
            symbol_or_symbols=TICKERS,
            timeframe=tf_day,
            start=start_date_daily,
            end=end_date,
            feed="iex"
        )
        bars_daily = client.get_stock_bars(request_daily).df
        if not bars_daily.empty:
            bars_daily.reset_index(inplace=True)
            bars_daily.to_csv(DAILY_FILE, index=False)
            print(f"Successfully saved {len(bars_daily)} daily rows to {os.path.basename(DAILY_FILE)}")
        else:
            print("Received empty Daily DataFrame from Alpaca.")
    except Exception as e:
        safe_msg = repr(e).encode("ascii", "replace").decode("ascii")
        print(f"Error fetching daily bars: {safe_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data for backtesting.")
    parser.add_argument("--days", type=int, default=60, help="Number of days back to fetch intraday data.")
    parser.add_argument("--start", type=str, default=None, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", type=str, default=None, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    load_data(days_back=args.days, start_str=args.start, end_str=args.end)
