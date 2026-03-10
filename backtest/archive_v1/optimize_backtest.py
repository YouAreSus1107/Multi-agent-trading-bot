import os
import sys
import random
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.run_backtest import run_backtest

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "optimization_results.csv")

def get_random_params():
    long_hybrid = random.choice([55, 60, 65, 70, 75, 80])
    return {
        "long_hybrid": long_hybrid,
        "long_exec": random.choice([20, 30, 40, 50]),
        "long_vwap": random.choice([-1.0, -1.5, -2.0]),
        
        "short_hybrid": 100 - long_hybrid,
        "short_exec": random.choice([15, 20, 25, 30]),
        "short_vwap": random.choice([1.0, 1.5, 2.0]),
        
        "stop_r": random.choice([1.0, 1.5, 2.0, 2.5]),
        "target_r": random.choice([2.0, 3.0, 4.0, 5.0]),
        
        "trail_1r": random.choice([True, False]),
        "trail_2r": random.choice([True, False]),
        "trail_activation_r": random.choice([0.5, 1.0, 1.5, 2.0]),
        "trailing_distance": random.choice([1.0, 1.5, 2.0])
    }

def run_optimization():
    print("Starting Infinite Random Search Optimizer for Backtester...")
    print("Results will be appended to the local optimization_results.csv file")
    print("Press Ctrl+C to stop at any time.\n")
    
    # Initialize CSV if it doesn't exist
    file_exists = os.path.isfile(RESULTS_FILE)
    if not file_exists:
        with open(RESULTS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "avg_pnl", "win_rate", "total_trades", "wins", "losses", 
                             "long_hybrid", "long_exec", "long_vwap", "short_hybrid", "short_exec", "short_vwap",
                             "stop_r", "target_r", "trail_1r", "trail_2r", "trail_activation_r", "trailing_distance"])

    best_pnl = -float('inf')
    iteration = 0
    
    try:
        while True:
            iteration += 1
            params = get_random_params()
            
            print(f"[{iteration}] Testing params: {params} ...", end="", flush=True)
            
            # Run backtest silently
            res = run_backtest(params=params, quiet=True)
            avg_pnl = res["avg_pnl"]
            
            print(f" | Avg PnL: {avg_pnl:.2f}% | Win Rate: {res['win_rate']:.2f}% | Trades: {res['total_trades']}")
            
            if avg_pnl > best_pnl:
                best_pnl = avg_pnl
                print(f"   *** NEW BEST: {best_pnl:.2f}% ***")
            
            # Append to CSV line by line
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(avg_pnl, 4), round(res["win_rate"], 2), res["total_trades"], res["wins"], res["losses"],
                    params["long_hybrid"], params["long_exec"], params["long_vwap"],
                    params["short_hybrid"], params["short_exec"], params["short_vwap"],
                    params["stop_r"], params["target_r"], params["trail_1r"], params["trail_2r"], 
                    params["trail_activation_r"], params["trailing_distance"]
                ])
                
    except KeyboardInterrupt:
        print("\nOptimization Interrupted by User. Exiting gracefully.")
        print(f"Best PnL found this session: {best_pnl:.2f}%")

if __name__ == "__main__":
    run_optimization()
