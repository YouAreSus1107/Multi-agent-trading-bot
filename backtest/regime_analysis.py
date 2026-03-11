"""
Regime Accuracy Analysis — Head-to-Head V2 vs V3
================================================
Compares the old HMA-Kahlman model against the new Multi-Factor State Machine.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader_v2 import parse_sp500_daily, SPY_TICKER
from backtest.indicators import compute_hma_kahlman_regime
from backtest.regime_v3 import MultiFactorRegimeModel, HMMRegimeModel

REPORT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORT_DIR, "regime_analysis.txt")

def analyze_model_performance(period_df, model_col, p_func):
    """Calculates accuracy and returns for a specific regime column"""
    regimes = period_df[model_col].unique()
    
    p_func(f"\n  Breakdown by {model_col} state:")
    p_func(f"  {'-'*65}")
    p_func(f"  {'State':<25} | {'Days':<5} | {'1d Acc':<8} | {'5d Acc':<8} | {'Avg 5d Ret':<10}")
    p_func(f"  {'-'*65}")
    
    for state in sorted(regimes):
        state_days = period_df[period_df[model_col] == state]
        days = len(state_days)
        if days == 0: continue
            
        # Directional Accuracy: 
        # Bull/Chop expects UP (>0), Bear expects DOWN (<0)
        # We classify Chop as neutral-to-up for this strict binary check, 
        # or we could just measure "absolute return < X%" for chop.
        # Keeping it simple: Bull=Up is win, Bear=Down is win. Chop ignored for directional acc.
        if state in ['bull', 'bull_trend']:
            acc_1d = (state_days['next_1d_return'] > 0).mean() * 100
            acc_5d = (state_days['next_5d_return'] > 0).mean() * 100
        elif state in ['bear', 'high_volatility_stress']:
            acc_1d = (state_days['next_1d_return'] < 0).mean() * 100
            acc_5d = (state_days['next_5d_return'] < 0).mean() * 100
        else:
            # For chop, "accuracy" is less about direction and more about low volatility/mean reversion
            # We'll just show the win-rate (pct of days > 0)
            acc_1d = (state_days['next_1d_return'] > 0).mean() * 100
            acc_5d = (state_days['next_5d_return'] > 0).mean() * 100
            
        avg_5d = state_days['next_5d_return'].mean()
        
        dir_marker = " (Down)" if state in ['bear', 'high_volatility_stress'] else " (Up)  "
        if state == 'chop': dir_marker = " (Flat)"
        
        p_func(f"  {state:<25} | {days:<5} | {acc_1d:>7.1f}%{dir_marker} | {acc_5d:>7.1f}%{dir_marker} | {avg_5d:>+9.3f}%")
        
    # Transition count
    transitions = (period_df[model_col] != period_df[model_col].shift(1)).sum() - 1
    p_func(f"\n  Total Transitions (Flips): {transitions}")
    
    # Strategy simulation
    # Long bias filter: Long only during bull/bull_trend. Cash (flat) otherwise.
    period_valid = period_df.dropna(subset=['next_1d_return'])
    def strat_logic(row):
        state = row[model_col]
        if state in ['bull', 'bull_trend']: return row['next_1d_return']
        return 0.0 # flat for bear, chop, high_volatility_stress
        
    strat_returns = period_valid.apply(strat_logic, axis=1)
    cum_ret = (1 + strat_returns / 100).cumprod().iloc[-1] if len(strat_returns) > 0 else 1
    p_func(f"  Simulated Strategy (Long Only): {(cum_ret - 1)*100:+.2f}%")


SP_DAILY_10Y = os.path.join(os.path.dirname(__file__), "data", "SnP_daily_update.csv")


def load_snp_index_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Build an equal-weighted S&P 500 OHLCV index from the multi-ticker CSV.
    Returns a DataFrame with columns [open, high, low, close, volume] indexed by date.
    """
    print(f"  Loading 10-year S&P data from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

    # Skip the first two metadata rows (Ticker / Date labels)
    df = df.iloc[2:].copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]

    # Convert all numeric values
    df = df.apply(pd.to_numeric, errors='coerce')

    result = pd.DataFrame(index=df.index)
    for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if field in df.columns.get_level_values(0):
            result[field.lower()] = df[field].mean(axis=1)  # equal-weight average

    result = result.dropna(how='all')
    print(f"  Built S&P index: {len(result)} trading days ({result.index[0].date()} to {result.index[-1].date()})")
    return result


def run_regime_analysis(start_year: str = "2022", end_year: str = "2023"):
    lines = []
    def p(msg=""):
        print(msg)
        lines.append(msg)
        
    p("=" * 80)
    p("REGIME MODEL HEAD-TO-HEAD: HMA-Kahlman vs HMM (Article Method)")
    p("=" * 80)
    
    # Load SPY for evaluation and HMA-Kahlman baseline
    all_tickers = parse_sp500_daily()
    if SPY_TICKER not in all_tickers:
        p("ERROR: SPY not found in dataset")
        return
    spy = all_tickers[SPY_TICKER].copy()
    
    # --- Model 1: V2 HMA-Kahlman on SPY ---
    v2_df = compute_hma_kahlman_regime(spy['close'], hk_length=14, kalman_gain=0.7, buffer_days=5)
    spy['v2_regime'] = v2_df['regime']
    
    # --- Model 2: Article-faithful HMM trained on 10-year S&P index ---
    # The model fetches VIX/HYG/LQD internally via yfinance and caches to disk.
    snp_index = load_snp_index_from_csv(SP_DAILY_10Y)
    print("\n  Fitting article-faithful HMM on 10-year S&P index...")
    hmm_model = HMMRegimeModel()
    hmm_df = hmm_model.compute_regime(snp_index)

    # Map regime from equal-weight index onto SPY by date alignment
    spy.index = pd.to_datetime(spy.index)
    hmm_df.index = pd.to_datetime(hmm_df.index)
    spy['hmm_regime'] = hmm_df['regime'].reindex(spy.index, method='ffill').fillna('bull')

    # Print diagnostics from HMM
    if hmm_model._diagnostics is not None:
        p("\n  HMM State Diagnostics (from training on 10Y S&P index):")
        p(f"  {'St':<4} {'Days':<6} {'AnnRet%':<10} {'AnnVol%':<10} {'AvgDD%':<10} {'Label'}")
        p(f"  {'-'*52}")
        for _, row in hmm_model._diagnostics.iterrows():
            lbl = hmm_model._state_map.get(int(row['state']), '?')
            p(f"  {int(row['state']):<4} {int(row['obs_count']):<6} "
              f"{row['ann_ret_%']:<10.2f} {row['ann_vol_%']:<10.2f} "
              f"{row['avg_drawdown_%']:<10.2f} {lbl}")

    # Compute forward returns on SPY
    spy['next_1d_return'] = spy['close'].pct_change(1).shift(-1) * 100
    spy['next_5d_return'] = spy['close'].pct_change(5).shift(-5) * 100
    
    start_dt = f"{start_year}-01-01"
    end_dt = f"{end_year}-12-31"
    period = spy.loc[start_dt:end_dt].copy()
    
    if len(period) == 0:
        p(f"No data for {start_dt} to {end_dt}")
        return
        
    bh_cum = (1 + period['next_1d_return'].dropna() / 100).cumprod().iloc[-1]
    
    p(f"\nPERIOD: {start_dt} to {end_dt} ({len(period)} trading days)")
    p(f"Buy & Hold SPY Return:   {(bh_cum - 1)*100:+.2f}%")
    
    p(f"\n{'='*80}")
    p("MODEL 1: HMA-KAHLMAN (Buffer=5)")
    p(f"{'='*80}")
    analyze_model_performance(period, 'v2_regime', p)
    
    p(f"\n{'='*80}")
    p("MODEL 2: HMM (Article Method) — PCA + BIC K-Selection + 27 Features")
    p("Trained on 10-Year S&P 500 Equal-Weighted Index + VIX/HYG/LQD")
    p(f"{'='*80}")
    analyze_model_performance(period, 'hmm_regime', p)

    
    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    p(f"\nReport saved to {os.path.basename(REPORT_FILE)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Head-to-head regime model comparison")
    parser.add_argument("--start", type=str, default="2022")
    parser.add_argument("--end", type=str, default="2023")
    args = parser.parse_args()
    
    run_regime_analysis(start_year=args.start, end_year=args.end)
