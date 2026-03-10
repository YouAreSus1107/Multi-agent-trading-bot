import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache_5m")

def check_fcx():
    file_path = os.path.join(CACHE_DIR, "FCX_2023-01-10.parquet")
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # Convert index to correct US/Eastern 
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('US/Eastern')
            
        print("--- FCX 2023-01-10 FULL DAY ---")
        pd.set_option('display.max_rows', None)
        print(df[['open', 'high', 'low', 'close']])
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    check_fcx()
