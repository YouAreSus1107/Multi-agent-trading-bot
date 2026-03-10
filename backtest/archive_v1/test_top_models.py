import os
import sys
import pandas as pd
import time
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.run_backtest import run_backtest
from backtest.fetch_datasets import PERIODS

def _run_single_backtest(params, hist_file, daily_file, return_dict):
    """Wrapper function to allow multiprocessing timeouts."""
    try:
        res = run_backtest(params=params, quiet=True, data_file=hist_file, daily_file=daily_file)
        return_dict['res'] = res
    except Exception as e:
        return_dict['error'] = str(e)

def main():
    print("====== Multi-Environment Top Model Cross-Validator ======\n")
    
    import argparse
    parser = argparse.ArgumentParser(description="Cross Validate Models.")
    parser.add_argument("--input", type=str, default="top_10_models.csv", help="Input CSV file name")
    args = parser.parse_args()

    input_csv = os.path.join(os.path.dirname(__file__), args.input)
    output_name = f"cv_{args.input}"
    output_csv = os.path.join(os.path.dirname(__file__), output_name)
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}. Please run top_models.py first.")
        return

    df = pd.read_csv(input_csv)
    if df.empty:
        print("The input CSV is empty.")
        return

    print(f"Testing {len(df)} models across {len(PERIODS)} different historical environments...\n")

    results = []

    for index, row in df.iterrows():
        print(f"--- Testing Model {index+1}/{len(df)} [Category: {row.get('category', 'Unknown')}] ---")
        
        # Extract params
        params = {
            "long_hybrid": int(row["long_hybrid"]),
            "long_exec": int(row["long_exec"]),
            "long_vwap": float(row["long_vwap"]),
            "short_hybrid": int(row["short_hybrid"]),
            "short_exec": int(row["short_exec"]),
            "short_vwap": float(row["short_vwap"]),
            "stop_r": float(row["stop_r"]),
            "target_r": float(row["target_r"]),
            "trail_1r": str(row["trail_1r"]).lower() == 'true',
            "trail_2r": str(row["trail_2r"]).lower() == 'true',
            "trail_activation_r": float(row["trail_activation_r"]),
            "trailing_distance": float(row["trailing_distance"]),
            "use_mtf": str(row.get("use_mtf", "False")).lower() == 'true'
        }
        
        total_pnl_sum = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for p in PERIODS:
            hist_file = os.path.join(data_dir, f"historical_bars_{p['start']}_{p['end']}.csv")
            daily_file = os.path.join(data_dir, f"daily_bars_{p['start']}_{p['end']}.csv")
            
            if not os.path.exists(hist_file) or not os.path.exists(daily_file):
                print(f"  [!] Missing datasets for {p['name']}. Skipping this period.")
                continue

            # Run backtest for this specific environment via process to enforce timeout
            print(f"  -> {p['name']:<35} | ", end="", flush=True)
            
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p_test = multiprocessing.Process(target=_run_single_backtest, args=(params, hist_file, daily_file, return_dict))
            
            start_t = time.time()
            p_test.start()
            p_test.join(240) # STRICT 4 MINUTE TIMEOUT PER BACKTEST
            
            if p_test.is_alive():
                print(f"[FAILED] Hung! Terminating after {int(time.time() - start_t)}s", flush=True)
                p_test.terminate()
                p_test.join()
                continue
                
            if 'error' in return_dict:
                print(f"[CRASHED] {return_dict['error']}", flush=True)
                continue
                
            res = return_dict.get('res')
            if not res:
                print("[FAILED] Null response", flush=True)
                continue
                
            pnl = res['avg_pnl']
            trades = res['total_trades']
            wins = res['wins']
            losses = res['losses']
            
            print(f"PnL: {pnl:>6.2f}% | Trades: {trades:>3}   (Took {int(time.time() - start_t)}s)", flush=True)
            
            # Aggregate stats
            total_trades += trades
            total_wins += wins
            total_losses += losses
            total_pnl_sum += (pnl * trades)
            
        if total_trades > 0:
            overall_pnl = total_pnl_sum / total_trades
            overall_wr = (total_wins / total_trades) * 100
        else:
            overall_pnl = 0
            overall_wr = 0

        print(f"  => OVERALL: PnL {overall_pnl:.2f}% | Win Rate {overall_wr:.2f}% | Trades {total_trades}\n")
        
        # Save aggregated robust stats back
        result_row = row.to_dict()
        result_row["robust_avg_pnl"] = round(overall_pnl, 4)
        result_row["robust_win_rate"] = round(overall_wr, 2)
        result_row["robust_trades"] = total_trades
        result_row["robust_wins"] = total_wins
        result_row["robust_losses"] = total_losses
        
        results.append(result_row)

    # Convert results back to dataframe and sort by robust PnL
    df_results = pd.DataFrame(results)
    
    # Sort robust PnL descending
    df_results = df_results.sort_values(by="robust_avg_pnl", ascending=False)
    
    df_results.to_csv(output_csv, index=False)
    
    print("=========================================================")
    print(f"Cross-Validation Complete! Saved to {output_csv}")
    print("Top 3 Most Robust Models Overall:")
    for i, (_, r) in enumerate(df_results.head(3).iterrows()):
        print(f" #{i+1} | {r.get('category', '')} | PnL: {r['robust_avg_pnl']:.2f}% | WR: {r['robust_win_rate']:.2f}%")

if __name__ == "__main__":
    main()
