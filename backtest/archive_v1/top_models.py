import pandas as pd
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract top X models with and without trailing stops.")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of top models to fetch for each category.")
    
    # Parse args, allowing the user's custom "--10" format
    args, unknown = parser.parse_known_args()
    
    top_n = args.top
    for arg in unknown:
        if arg.startswith("--") and arg[2:].isdigit():
            top_n = int(arg[2:])
            break
        elif arg.startswith("-") and arg[1:].isdigit():
            top_n = int(arg[1:])
            break

    input_csv = os.path.join(os.path.dirname(__file__), "optimization_results.csv")
    output_csv = os.path.join(os.path.dirname(__file__), f"top_{top_n}_models.csv")

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} does not exist.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("The optimization results file is empty.")
        return

    # Check for trailing stops: if either trail_1r or trail_2r is True
    # Casting to string handles both boolean and string literal cases
    condition_trailing = (df['trail_1r'].astype(str).str.lower() == 'true') | (df['trail_2r'].astype(str).str.lower() == 'true')
    
    df_trailing = df[condition_trailing]
    df_no_trailing = df[~condition_trailing]

    # Sort by avg_pnl descending
    df_trailing_sorted = df_trailing.sort_values(by="avg_pnl", ascending=False).head(top_n)
    df_no_trailing_sorted = df_no_trailing.sort_values(by="avg_pnl", ascending=False).head(top_n)

    # Insert a category column at the front to clarify
    df_trailing_sorted.insert(0, "category", "With Trailing")
    df_no_trailing_sorted.insert(0, "category", "Without Trailing")

    # Combine into one dataframe
    df_combined = pd.concat([df_trailing_sorted, df_no_trailing_sorted], ignore_index=True)

    # Save to the separate CSV
    df_combined.to_csv(output_csv, index=False)
    
    print(f"Successfully extracted the top {top_n} models for each category.")
    print(f"Saved to: top_{top_n}_models.csv\n")
    
    print("--- Summary ---")
    
    if not df_trailing_sorted.empty:
        best_trail = df_trailing_sorted.iloc[0]
        print(f"Best 'With Trailing'    -> Avg PnL: {best_trail['avg_pnl']:.4f}% | Win Rate: {best_trail['win_rate']:.2f}% | Trades: {best_trail['total_trades']}")
    else:
        print("No 'With Trailing' models found.")
        
    if not df_no_trailing_sorted.empty:
        best_no_trail = df_no_trailing_sorted.iloc[0]
        print(f"Best 'Without Trailing' -> Avg PnL: {best_no_trail['avg_pnl']:.4f}% | Win Rate: {best_no_trail['win_rate']:.2f}% | Trades: {best_no_trail['total_trades']}")
    else:
        print("No 'Without Trailing' models found.")

if __name__ == "__main__":
    main()
