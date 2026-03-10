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
from backtest.regime_v3 import MultiFactorRegimeModel

REPORT_FILE = os.path.join(os.path.dirname(__file__), "regime_analysis.txt")


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
            
        acc_1d = (state_days['next_1d_return'] > 0).mean() * 100
        acc_5d = (state_days['next_5d_return'] > 0).mean() * 100
        avg_5d = state_days['next_5d_return'].mean()
        
        p_func(f"  {state:<25} | {days:<5} | {acc_1d:>7.1f}% | {acc_5d:>7.1f}% | {avg_5d:>+9.3f}%")
        
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


def run_regime_analysis(start_year: str = "2022", end_year: str = "2023"):
    lines = []
    def p(msg=""):
        print(msg)
        lines.append(msg)
        
    p("=" * 80)
    p("REGIME MODEL HEAD-TO-HEAD: V2 (HMA) vs V3 (Multi-Factor)")
    p("=" * 80)
    
    # Load data
    all_tickers = parse_sp500_daily()
    if SPY_TICKER not in all_tickers:
        p("ERROR: SPY not found in dataset")
        return
    spy = all_tickers[SPY_TICKER].copy()
    
    # Compute V2 Regime (HMA-Kahlman with best param buffer=5)
    v2_df = compute_hma_kahlman_regime(spy['close'], hk_length=14, kalman_gain=0.7, buffer_days=5)
    spy['v2_regime'] = v2_df['regime']
    
    # Compute V3 Regime (Multi-Factor)
    v3_model = MultiFactorRegimeModel(min_dwell_days=3)
    v3_df = v3_model.compute_regime(spy)
    spy['v3_regime'] = v3_df['regime']
    
    # Compute forward returns
    spy['next_1d_return'] = spy['close'].pct_change(1).shift(-1) * 100
    spy['next_5d_return'] = spy['close'].pct_change(5).shift(-5) * 100
    
    # Slice to analysis period
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
    p("MODEL 1: V2 HMA-KAHLMAN (Buffer=5)")
    p(f"{'='*80}")
    analyze_model_performance(period, 'v2_regime', p)
    
    p(f"\n{'='*80}")
    p("MODEL 2: V3 MULTI-FACTOR STATE MACHINE")
    p("Features: Trend Alignment, Realized Volatility, Kaufman Efficiency")
    p(f"{'='*80}")
    analyze_model_performance(period, 'v3_regime', p)
    
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
