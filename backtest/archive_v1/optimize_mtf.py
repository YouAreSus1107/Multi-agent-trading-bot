import os
import sys
import random
import csv
import time
import pandas as pd
from datetime import datetime

os.environ["DISABLE_FILE_LOGGING"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.run_backtest import run_backtest
from backtest.mock_market_service import MockMarketService
from backtest.fetch_datasets import PERIODS

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "mtf_optimization_results.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CSV_HEADER = [
    "timestamp", "strategy", "profit_factor", "win_rate", "total_trades", "avg_pnl",
    "bull_pnl", "bear_pnl", "chop_pnl", "min_regime_pnl", 
    "gross_profit", "gross_loss",
    "long_hybrid", "long_exec", "long_vwap", "short_hybrid", "short_exec", "short_vwap",
    "long_delta_min", "short_delta_max", "long_bounce_pct", "short_bounce_pct",
    "stop_r", "target_r"
]


def get_random_mtf_params():
    strategy_choices = ["trend", "strict", "pullback"]
    choice = random.choice(strategy_choices)
    
    if choice == "trend":
        params = {
            "category": "MTF Trend Follower (Loose Entry)",
            "strategy_type": "trend",
            "long_hybrid": random.randint(60, 75),
            "long_exec": random.randint(25, 45),
            "long_vwap": round(random.uniform(0.0, 1.0), 1),
            "short_hybrid": random.randint(25, 40),
            "short_exec": random.randint(25, 45),
            "short_vwap": round(random.uniform(-1.0, 0.0), 1),
            # Not used by trend
            "long_delta_min": 0.0,
            "short_delta_max": 0.0,
            "long_bounce_pct": 0.0,
            "short_bounce_pct": 0.0,
        }
    elif choice == "strict":
        params = {
            "category": "MTF Strict Execution (Momentum Breakout)",
            "strategy_type": "strict",
            "long_hybrid": random.randint(55, 72),
            "long_exec": random.randint(30, 48),
            "long_vwap": 0.0,  # Not used 
            "short_hybrid": random.randint(28, 45),
            "short_exec": random.randint(30, 48),
            "short_vwap": 0.0,  # Not used
            "long_delta_min": round(random.uniform(0.55, 0.70), 2),
            "short_delta_max": round(random.uniform(0.30, 0.45), 2),
            "long_bounce_pct": 0.0,
            "short_bounce_pct": 0.0,
        }
    else:
        params = {
            "category": "MTF Pullback Buyer (Confirmed Reversal)",
            "strategy_type": "pullback",
            "long_hybrid": 0.0,  # Not used 
            "long_exec": random.randint(15, 35),
            "long_vwap": round(random.uniform(-3.0, -1.0), 1),
            "short_hybrid": 0.0,  # Not used
            "short_exec": random.randint(30, 50),
            "short_vwap": round(random.uniform(1.0, 3.0), 1),
            "long_delta_min": 0.0,
            "short_delta_max": 0.0,
            "long_bounce_pct": round(random.uniform(0.55, 0.75), 2),
            "short_bounce_pct": round(random.uniform(0.25, 0.45), 2),
        }
        
    # Shared risk parameters
    params.update({
        "stop_r": random.choice([1.0, 1.25, 1.5, 1.75, 2.0]),
        "target_r": random.choice([2.0, 2.5, 3.0, 3.5, 4.0]),
        "trail_1r": False,
        "trail_2r": False,
        "trail_activation_r": 0.0,
        "trailing_distance": 0.0,
        "use_mtf": True
    })
    
    return params


def format_param_line(params):
    """Format a readable parameter summary for console output."""
    st = params["strategy_type"]
    if st == "trend":
        return f"L: {params['long_hybrid']}/{params['long_exec']}/{params['long_vwap']} | S: {params['short_hybrid']}/{params['short_exec']}/{params['short_vwap']} | RR: {params['stop_r']}:{params['target_r']}"
    elif st == "strict":
        return f"L: H{params['long_hybrid']}/E{params['long_exec']}/D>={params['long_delta_min']} | S: H{params['short_hybrid']}/E{params['short_exec']}/D<={params['short_delta_max']} | RR: {params['stop_r']}:{params['target_r']}"
    else:
        return f"L: VWAP<{params['long_vwap']}/E{params['long_exec']}/B>={params['long_bounce_pct']} | S: VWAP>{params['short_vwap']}/E{params['short_exec']}/B<={params['short_bounce_pct']} | RR: {params['stop_r']}:{params['target_r']}"


def run_optimization():
    print("====== MTF Multi-Environment Infinite Optimizer (v2 — Rearchitected) ======")
    print("Strategies: Trend Follower | Strict Execution (Breakout) | Pullback Buyer (Reversal)")
    print("Ranking by Profit Factor. Press Ctrl+C to stop.\n")
    
    # ======= PRE-LOAD ALL DATASETS INTO MEMORY ONCE =======
    print("Pre-loading all historical datasets into memory (one-time cost)...")
    preloaded_services = []
    for p in PERIODS:
        hist_file = os.path.join(DATA_DIR, f"historical_bars_{p['start']}_{p['end']}.csv")
        daily_file = os.path.join(DATA_DIR, f"daily_bars_{p['start']}_{p['end']}.csv")
        
        if not os.path.exists(hist_file) or not os.path.exists(daily_file):
            print(f"  [!] Missing datasets for {p['name']}. Skipping.")
            continue
        
        t0 = time.time()
        ms = MockMarketService(current_time=pd.Timestamp.now(tz='UTC'), data_file=hist_file, daily_file=daily_file)
        print(f"  Loaded {p['name']:45} ({time.time() - t0:.1f}s)")
        preloaded_services.append({"period": p, "service": ms})
    
    if not preloaded_services:
        print("No datasets loaded! Exiting.")
        return
        
    print(f"\nAll {len(preloaded_services)} datasets ready. Starting optimization loop.\n")
    
    # Initialize CSV if needed
    file_exists = os.path.isfile(RESULTS_FILE)
    if not file_exists:
        with open(RESULTS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    best_pf = -float('inf')
    iteration = 0
    
    try:
        while True:
            iteration += 1
            params = get_random_mtf_params()
            
            print(f"[{iteration}] Testing: {params['category']}")
            print(f"        Params => {format_param_line(params)}")
            
            total_trades = 0
            total_wins = 0
            total_losses = 0
            total_pnl_sum = 0
            
            # Regime tracking
            regime_pnl = {"Bull": 0.0, "Bear": 0.0, "Chop": 0.0}
            regime_trades = {"Bull": 0, "Bear": 0, "Chop": 0}
            
            total_gross_profit = 0
            total_gross_loss = 0
            
            failed_env = False
            
            for entry in preloaded_services:
                p = entry["period"]
                ms = entry["service"]
                
                print(f"  -> {p['name']:38} | ", end="", flush=True)
                
                start_t = time.time()
                try:
                    res = run_backtest(params=params, quiet=True, market_service=ms)
                except Exception as e:
                    print(f"[CRASHED] {e}", flush=True)
                    failed_env = True
                    break
                    
                elapsed = int(time.time() - start_t)
                
                if not res:
                    print(f"[FAILED] Null response", flush=True)
                    failed_env = True
                    break
                    
                print(f"PnL: {res['avg_pnl']:6.2f}% | Trades: {res['total_trades']:<4}  (Took {elapsed}s)")
                    
                total_trades += res['total_trades']
                total_wins += res['wins']
                total_losses += res['losses']
                total_pnl_sum += (res['avg_pnl'] * res['total_trades'])
                total_gross_profit += res.get('gross_profit', 0)
                total_gross_loss += res.get('gross_loss', 0)
                
                # Attribute to regime
                if "Bull" in p['name']:
                    regime_pnl["Bull"] += res['avg_pnl'] * res['total_trades']
                    regime_trades["Bull"] += res['total_trades']
                elif "Bear" in p['name']:
                    regime_pnl["Bear"] += res['avg_pnl'] * res['total_trades']
                    regime_trades["Bear"] += res['total_trades']
                elif "Chop" in p['name'] or "Sideways" in p['name']:
                    regime_pnl["Chop"] += res['avg_pnl'] * res['total_trades']
                    regime_trades["Chop"] += res['total_trades']
                
            if failed_env:
                print("--- Skipped recording due to environment failure ---\n")
                continue
                
            if total_trades > 0:
                overall_pnl = total_pnl_sum / total_trades
                overall_wr = (total_wins / total_trades) * 100
                profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else total_gross_profit
                
                bull_avg = regime_pnl["Bull"] / regime_trades["Bull"] if regime_trades["Bull"] > 0 else 0
                bear_avg = regime_pnl["Bear"] / regime_trades["Bear"] if regime_trades["Bear"] > 0 else 0
                chop_avg = regime_pnl["Chop"] / regime_trades["Chop"] if regime_trades["Chop"] > 0 else 0
                min_regime = min(bull_avg, bear_avg, chop_avg)
            else:
                overall_pnl = 0
                overall_wr = 0
                profit_factor = 0
                bull_avg = bear_avg = chop_avg = min_regime = 0
                
            print(f"  => RESULT: Profit Factor {profit_factor:.2f} | PnL {overall_pnl:.2f}% | WR {overall_wr:.2f}% | Trades {total_trades}")
            print(f"     Regimes: Bull {bull_avg:.2f}% | Bear {bear_avg:.2f}% | Chop {chop_avg:.2f}%  (Min: {min_regime:.2f}%)")
            
            if profit_factor > best_pf and total_trades > 30:
                best_pf = profit_factor
                print(f"   *** NEW BEST PF: {best_pf:.2f} ***")
            print()
            
            # Multi-process collision handling for writing results
            row_data = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                params["category"], round(profit_factor, 2), round(overall_wr, 2), total_trades, round(overall_pnl, 4),
                round(bull_avg, 4), round(bear_avg, 4), round(chop_avg, 4), round(min_regime, 4),
                round(total_gross_profit, 2), round(total_gross_loss, 2),
                params["long_hybrid"], params["long_exec"], params["long_vwap"],
                params["short_hybrid"], params["short_exec"], params["short_vwap"],
                params.get("long_delta_min", ""), params.get("short_delta_max", ""),
                params.get("long_bounce_pct", ""), params.get("short_bounce_pct", ""),
                params["stop_r"], params["target_r"]
            ]
            
            written = False
            retries = 10
            while not written and retries > 0:
                try:
                    with open(RESULTS_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)
                    written = True
                except PermissionError:
                    # Very slight sleep to let other terminal release the file lock
                    time.sleep(random.uniform(0.1, 0.5))
                    retries -= 1
                
    except KeyboardInterrupt:
        print("\n\nOptimization Interrupted by User.")
        
    # Sort the CSV by Profit Factor before exiting
    try:
        if os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)
            if not df.empty:
                df = df.sort_values(by="profit_factor", ascending=False)
                df.to_csv(RESULTS_FILE, index=False)
                print(f"Sorted {len(df)} results in {os.path.basename(RESULTS_FILE)} by Profit Factor.")
    except Exception as e:
        print(f"Warning: Could not sort results file: {e}")
        
    print(f"Best Profit Factor this session: {best_pf:.2f}")
    print("Exiting gracefully.")

if __name__ == "__main__":
    run_optimization()
