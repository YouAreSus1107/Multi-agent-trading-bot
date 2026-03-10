import pandas as pd
import re
from datetime import datetime

log_file = "results_v2.log"

with open(log_file, "r") as f:
    lines = f.readlines()

trades = []
current_date = None

for line in lines:
    if line.startswith("  [2"):
        match = re.search(r"\[(.*?)\] (LONG|STOP|EOD CLOSE|TARGET) ([A-Z]+)", line)
        if match:
            date_str, action, ticker = match.groups()
            
            pnl_match = re.search(r"PnL: ([+-][0-9.]+)%", line)
            pnl = float(pnl_match.group(1)) if pnl_match else None
            
            if action in ["STOP", "EOD CLOSE", "TARGET"]:
                trades.append({
                    "date": date_str,
                    "action": action,
                    "ticker": ticker,
                    "pnl": pnl
                })

df = pd.DataFrame(trades)
if not df.empty:
    print(f"Total Closed Trades: {len(df)}")
    print(df['action'].value_counts())
    print("\n--- PnL by Exit Type ---")
    print(df.groupby('action')['pnl'].describe()[['count', 'mean', 'min', 'max']])
    print("\n--- Daily PnL Volatility ---")
    daily_pnl = df.groupby('date')['pnl'].sum()
    print(f"Worst Day: {daily_pnl.min():.2f}%")
    print(f"Best Day: {daily_pnl.max():.2f}%")
    print(f"Average Daily PnL: {daily_pnl.mean():.2f}%")
    print(f"Win Rate: {(df['pnl'] > 0).mean()*100:.1f}%")
    
    eod = df[df['action'] == 'EOD CLOSE']
    print(f"\nEOD Close Win Rate: {(eod['pnl'] > 0).mean()*100:.1f}%")
    print(f"EOD Close Avg PnL: {eod['pnl'].mean():.2f}%")
    
    stops = df[df['action'] == 'STOP']
    print(f"\nStop Avg Loss: {stops['pnl'].mean():.2f}%")
    
    # Calculate consecutive losses
    loss_streak = 0
    max_streak = 0
    for pnl in df['pnl']:
        if pnl < 0:
            loss_streak += 1
            max_streak = max(max_streak, loss_streak)
        else:
            loss_streak = 0
            
    print(f"Max Consecutive Losses: {max_streak}")
    
    # Calculate R:R
    avg_win = df[df['pnl'] > 0]['pnl'].mean()
    avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean())
    print(f"\nAverage Win: +{avg_win:.2f}%")
    print(f"Average Loss: -{avg_loss:.2f}%")
    print(f"Reward:Risk Ratio: {(avg_win/max(avg_loss, 0.01)):.2f}")
