import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest_v2 import run_backtest_v2

if __name__ == "__main__":
    print("Starting debug run...")
    start_time = time.time()
    res = run_backtest_v2(
        start_date="2023-01-03", 
        end_date="2023-01-04", 
        top_n=3, 
        fetch_5m=True, 
        verbose=True
    )
    print(f"Finished in {time.time() - start_time:.2f}s")
