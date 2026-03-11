"""
V2 Backtest Runner — 3-Layer Backtesting Architecture
======================================================
Ties together:
  Layer 1: Daily Universe Ranking (data_loader_v2.py)
  Layer 2: Regime Detection - HMA-Kahlman (indicators.py) 
  Layer 3: Intraday 5m Execution (this file)

Execution modes:
  Bull Regime → MTF Strict Long (Score>=56, Exec>=36, DR>=0.57)
  Bear Regime → RSI-VWAP Short (RSI>70, VWAP Z>2.0) + Inverse ETF Long
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import traceback
import logging

log = logging.getLogger(__name__)

# Load static S&P 500 sectors mapping
try:
    sector_path = os.path.join(os.path.dirname(__file__), 'sp500_sectors.json')
    with open(sector_path, 'r') as f:
        SP500_SECTORS = json.load(f)
except Exception as e:
    SP500_SECTORS = {}
    print(f"Warning: Failed to load sp500_sectors.json: {e}")

INVERSE_ETFS = ['PSQ', 'SQQQ', 'QID', 'SDS', 'SPXU', 'SH', 'DOG', 'DXD', 'SDOW']

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader_v2 import parse_sp500_daily, run_daily_selection, SPY_TICKER
from backtest.regime_v3 import HMMRegimeModel

# Production TA stack (exact same code as live trading)
from utils.indicators import compute_trend_score, compute_daily_returns, compute_sharpe_ratio
from utils.quant_engine import (
    precompute_quant_metrics_for_ticker,
    precompute_mtf_tsl_for_ticker,
    build_quant_metrics
)

LOG_FILE = os.path.join(os.path.dirname(__file__), "results_v2.log")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def load_strategy_params(filepath: str) -> dict:
    """
    Load dynamic strategy parameters from a CSV-like txt file.
    Format expected:
    long_hybrid,long_exec,long_vwap,short_hybrid,short_exec,short_vwap,long_delta_min,short_delta_max,long_bounce_pct,short_bounce_pct,stop_r,target_r
    Input: 5.0,33,0.0,45.0,39,0.0,0.58,0.43,0.0,0.0,1.5,3.5
    """
    default_params = {
        'long_hybrid': 56.0,
        'long_exec': 36.0,
        'long_vwap': 2.5,
        'short_hybrid': 45.0,
        'short_exec': 34.0,
        'short_vwap': -2.5,
        'long_delta_min': 0.55,
        'short_delta_max': 0.40,
        'long_bounce_pct': 0.0,
        'short_bounce_pct': 0.0,
        'stop_r': 1.5,
        'target_r': 3.0,
    }
    
    if not filepath or not os.path.exists(filepath):
        return default_params
        
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if len(lines) >= 2:
            keys = [k.strip() for k in lines[0].split(',')]
            val_line = lines[1]
            if val_line.lower().startswith("input:"):
                val_line = val_line.replace("Input:", "").replace("input:", "").strip()
            
            vals = [float(v.strip()) for v in val_line.split(',')]
            
            for k, v in zip(keys, vals):
                if k in default_params:
                    default_params[k] = v
        
        print(f"Loaded params from {filepath}: {default_params}")
        return default_params
    except Exception as e:
        print(f"Error loading param file {filepath}: {e}, using defaults.")
        return default_params


def evaluate_bull_entry(quant: dict, trend_score: float, sparm: dict) -> tuple[bool, str]:
    """
    MTF Strict Long entry conditions matching V1 'MTF Strict Execution (Momentum Breakout)':
      Best params from CSV: long_hybrid=56, long_exec=36, long_delta_min=0.57
    
    IMPORTANT: trend_score here is already scaled to 0-40 range (raw_score * 0.40),
    matching V1 mock_technical_analyst.py line 67:
        trend_score = round(raw_classic_score * 0.40, 1)
    Then in V1: hybrid_score = round(trend_score + exec_score)
    DO NOT multiply trend_score by 0.40 again.
    """
    # Compute exec_score from quant metrics (same logic as V1 mock_technical_analyst.py)
    exec_score = 30
    z = quant.get('vwap_zscore', 0)
    if z < -2.5: exec_score += 20
    elif z < -1.5: exec_score += 12
    elif z < -0.5: exec_score += 6
    elif z > 2.5: exec_score -= 20
    elif z > 1.5: exec_score -= 10
    elif z > 0.5: exec_score -= 3
    
    dr = quant.get('delta_ratio', 0.5)
    if dr > 0.65: exec_score += 25
    elif dr > 0.55: exec_score += 14
    elif dr < 0.35: exec_score -= 25
    elif dr < 0.45: exec_score -= 12
    
    if quant.get('smart_bounce'):
        exec_score += 15
    
    exec_score = max(0, min(60, exec_score))
    
    # Hybrid score: V1 formula is trend_score (already 40% of raw) + exec_score
    # trend_score is in 0-40 range, exec_score in 0-60 range -> max hybrid = 100
    hybrid_score = round(trend_score + exec_score)
    hybrid_score = max(0, min(100, hybrid_score))
    
    # MTF Gate (Removed per user request)
    mtf_total_pos = quant.get('mtf_total_pos', 0)
    
    # Dynamic MTF Strict Execution:
    # Requires hybrid_score >= long_hybrid
    # Requires exec_score >= long_exec
    # Requires delta_ratio >= long_delta_min
    # Requires vwap_zscore <= long_vwap
    # Optional bounce pct requirement if long_bounce_pct > 0
    last_buy_pct = quant.get('last_bar_buy_pct', 0.5)
    
    entry = (
        hybrid_score >= sparm['long_hybrid'] and 
        exec_score >= sparm['long_exec'] and 
        dr >= sparm['long_delta_min'] and 
        z <= sparm['long_vwap']
    )
    
    if sparm['long_bounce_pct'] > 0.0:
        entry = entry and (last_buy_pct >= sparm['long_bounce_pct'])
    
    return entry, f"Score={hybrid_score} Exec={exec_score} DR={dr:.2f} Z={z:.2f}"


def evaluate_bear_entry(quant: dict, trend_score: float, sparm: dict) -> tuple[bool, str]:
    """
    Mean Reversion Short Entry Logic for Bear Regimes:
    Target: Overextended pumps (RSI > 70, VWAP Z > 2.0) with fading momentum.
    """
    exec_score = 30
    
    # 1. Require extreme UPWARD extension (Mean Reversion)
    z = quant.get('vwap_zscore', 0)
    if z > 2.5: exec_score += 20
    elif z > 2.0: exec_score += 12
    elif z > 1.5: exec_score += 6
    # Penalize if it's already dumping
    elif z < -1.0: exec_score -= 20
    
    # 2. RSI Overbought Check
    rsi = quant.get('rsi_14', 50)
    if rsi > 75: exec_score += 15
    elif rsi > 70: exec_score += 10
    
    # 3. Delta Ratio (Wait for buyers to exhaust)
    dr = quant.get('delta_ratio', 0.5)
    if dr < 0.40: exec_score += 15  # Sellers taking control
    elif dr > 0.60: exec_score -= 20 # Buyers still pushing
    
    exec_score = max(0, min(60, exec_score))
    
    bear_trend = max(0.0, 40.0 - trend_score)
    hybrid_score = round(bear_trend + exec_score)
    hybrid_score = max(0, min(100, hybrid_score))
    
    short_entry = (
        hybrid_score >= sparm['short_hybrid'] and 
        exec_score >= sparm['short_exec'] and
        dr <= sparm['short_delta_max'] and
        z >= sparm['short_vwap'] # Note: Optimizer should now hunt for positive values (e.g., 2.0)
    )
    
    return short_entry, f"Score={hybrid_score} Exec={exec_score} DR={dr:.2f} Z={z:.2f} RSI={rsi:.1f}"


def evaluate_inverse_etf_entry(quant: dict, trend_score: float, sparm: dict) -> tuple[bool, str]:
    """
    Long entry on inverse ETFs during bear regime.
    Uses slightly relaxed MTF conditions based on sparm since these are directional hedges.
    """
    exec_score = 30
    z = quant.get('vwap_zscore', 0)
    if z < -2.5: exec_score += 20
    elif z < -1.5: exec_score += 12
    elif z < -0.5: exec_score += 6
    elif z > 2.5: exec_score -= 20
    elif z > 1.5: exec_score -= 10
    elif z > 0.5: exec_score -= 3
    
    dr = quant.get('delta_ratio', 0.5)
    if dr > 0.55: exec_score += 14
    elif dr > 0.65: exec_score += 25
    elif dr < 0.35: exec_score -= 25
    elif dr < 0.45: exec_score -= 12
    
    exec_score = max(0, min(60, exec_score))
    
    # Invert the trend score for inverse ETFs (they are bearish derivatives)
    bear_trend = max(0.0, 40.0 - trend_score)
    hybrid_score = round(bear_trend + exec_score)
    hybrid_score = max(0, min(100, hybrid_score))
    
    # Relaxed thresholds for inverse ETF longs compared to regular bearish entries (subtract 10 points)
    min_hybrid = max(45, sparm['short_hybrid'] - 12)  # Use relative to short_hybrid since inverted
    min_exec = max(30, sparm['short_exec'] - 10)
    max_dr = min(0.55, sparm['short_delta_max'] + 0.10)
    
    entry = (
        hybrid_score >= min_hybrid and 
        exec_score >= min_exec and 
        dr <= max_dr and 
        z >= sparm['short_vwap']
    )
    
    return entry, f"Score={hybrid_score} Exec={exec_score} DR={dr:.2f} Z={z:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# POSITION MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def manage_position(pos: dict, current_price: float, current_time, params: dict, current_mtf_tsl: float = None) -> tuple[bool, str, float]:
    """
    Check stop/target/trailing for a position using dynamic MTF TSL.
    Returns: (is_closed, close_type, pnl_pct)
    """
    is_long = pos['side'] == 'long'
    entry_price = pos['entry_price']
    atr = pos['atr_5m']
    stop_r = params.get('stop_r', 3.0)      
    target_r = params.get('target_r', 3.0)  
    leverage = pos.get('leverage', 1.0)
    
    is_morning_lock = current_time.hour == 9 or (current_time.hour == 10 and current_time.minute <= 30)
    circuit_breaker_loss_limit = -4.0
    
    dir_mult = 1.0 if is_long else -1.0
    open_pnl = ((current_price - entry_price) / entry_price * dir_mult) * 100 * leverage
    
    if is_morning_lock:
        if open_pnl <= circuit_breaker_loss_limit:
            return True, 'stop', open_pnl
        elif current_price >= pos['target'] if is_long else current_price <= pos['target']:
            return True, 'target', open_pnl
        return False, '', 0.0

    # --- DYNAMIC MTF TSL LOGIC ---
    if current_mtf_tsl is not None and not np.isnan(current_mtf_tsl):
        if is_long:
            if current_mtf_tsl > pos['stop']:
                pos['stop'] = current_mtf_tsl
        else:
            if current_mtf_tsl < pos['stop']:
                pos['stop'] = current_mtf_tsl

    # Standard Exits against the dynamic stop
    if is_long:
        if current_price <= pos['stop']:
            pnl = ((current_price - entry_price) / entry_price * leverage) * 100
            return True, 'stop', pnl
        elif current_price >= pos['target']:
            pnl = ((current_price - entry_price) / entry_price * leverage) * 100
            return True, 'target', pnl
    else:
        if current_price >= pos['stop']:
            pnl = ((entry_price - current_price) / entry_price * leverage) * 100
            return True, 'stop', pnl
        elif current_price <= pos['target']:
            pnl = ((entry_price - current_price) / entry_price * leverage) * 100
            return True, 'target', pnl
            
    return False, '', 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest_v2(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    top_n: int = 10,
    fetch_5m: bool = True,
    verbose: bool = True,
    always_long: bool = False,
    params: dict = None,
    sparm: dict = None
) -> dict:
    """
    Run the complete V2 3-layer backtest.
    
    1. Parse S&P 500 daily data
    2. For each trading day: rank universe, detect regime, execute on 5m bars
    3. Output performance report
    """
    if sparm is None:
        sparm = load_strategy_params(None)
        
    if params is None:
        params = {
            'stop_r': sparm.get('stop_r', 1.5),      # ATR multiplier for stop
            'target_r': sparm.get('target_r', 3.0),    # ATR multiplier for target
            'risk_per_trade': 0.05,  # 5% risk per trade
            'max_positions': 5,
            'max_hold_days': 0,      # 0 = strictly intraday (force close at EOD)
        }
    
    log_lines = []
    
    def log(msg):
        if verbose:
            print(msg)
        log_lines.append(msg)
    
    log("=" * 70)
    log("V2 BACKTEST ENGINE - 3-Layer Architecture")
    log(f"Period: {start_date} to {end_date}")
    log("=" * 70)
    
    # ── Step 1: Load S&P 500 daily data ──
    log("\n[Layer 1] Loading S&P 500 daily data...")
    all_tickers = parse_sp500_daily(fetch_yfinance=fetch_5m)
    log(f"  Loaded {len(all_tickers)} tickers")
    
    # ── Step 2: Run daily selection + regime detection ──
    log("\n[Layer 1+2] Running daily Universe Ranking + Regime Detection...")
    daily_snapshots = run_daily_selection(
        all_tickers,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        fetch_5m=fetch_5m,
        verbose=verbose
    )
    
    if not daily_snapshots:
        log("No trading days processed. Exiting.")
        return {}
    
    # ── Step 3: Intraday Execution ──
    log(f"\n[Layer 3] Running Intraday 5m Execution on {len(daily_snapshots)} trading days...")
    
    positions = []
    trade_history = []
    equity = 10000.0  # Starting capital
    equity_curve = []
    
    for day_idx, snapshot in enumerate(daily_snapshots):
        trade_date = snapshot['date']
        regime = snapshot['regime']
        top_picks = snapshot['top_tickers']
        intraday_data = snapshot.get('intraday_data', {})
        daily_stopped_tickers = set()  # Anti-revenge rule
        
        if not intraday_data:
            continue
        
        day_str = trade_date.strftime('%Y-%m-%d')
        
        # Check if held positions are missing intraday data (happens on regime/top-10 switch)
        held_tickers_missing = [p['ticker'] for p in positions if p['ticker'] not in intraday_data]
        if held_tickers_missing and fetch_5m:
            from backtest.data_loader_v2 import fetch_5m_bars_for_day
            extra = fetch_5m_bars_for_day(held_tickers_missing, trade_date.to_pydatetime() if hasattr(trade_date, 'to_pydatetime') else trade_date)
            intraday_data.update(extra)
        
        # Get all unique 5m timestamps across all tickers for this day
        all_timestamps = set()
        for ticker, bars_df in intraday_data.items():
            if bars_df is not None and len(bars_df) > 0:
                all_timestamps.update(bars_df.index.tolist())
        
        if not all_timestamps:
            continue
        
        sorted_times = sorted(all_timestamps)
        
        # Skip first 25% of timestamps (warmup for intraday indicators)
        warmup_cutoff = 16
        execution_times = sorted_times[warmup_cutoff:]
        
        if not execution_times:
            continue
        
        # Daily trend scores (computed once per day from daily data)
        # Uses the exact same production compute_trend_score (RSI+MACD+EMA+BB+Vol)
        daily_trend_scores = {}
        for pick in top_picks:
            ticker = pick['ticker']
            if ticker in all_tickers:
                tdf = all_tickers[ticker]
                # Shift slice back one day to prevent forward-looking data leakage
                slice_end = trade_date - pd.Timedelta(days=1)
                daily_slice = tdf.loc[:slice_end]
                
                if len(daily_slice) >= 26:
                    closes_list = daily_slice['close'].tolist()[-60:]
                    volumes_list = daily_slice['volume'].tolist()[-60:] if 'volume' in daily_slice.columns else None
                    highs_list = daily_slice['high'].tolist()[-60:] if 'high' in daily_slice.columns else None
                    lows_list = daily_slice['low'].tolist()[-60:] if 'low' in daily_slice.columns else None
                    classic = compute_trend_score(closes_list, volumes_list, highs_list, lows_list)
                    raw_trend = classic['score']  # 0-100
                    trend_score = round(raw_trend * 0.40, 1)  # Scale to 0-40, matching V1
                    daily_trend_scores[ticker] = trend_score
                else:
                    daily_trend_scores[ticker] = 20.0
        
        # Step through 5m candles
        for t_idx, ts in enumerate(execution_times):
            # 1. Manage existing positions
            for pos in list(positions):
                ticker = pos['ticker']
                if ticker not in intraday_data:
                    continue
                bars = intraday_data[ticker]
                if bars is None or len(bars) == 0:
                    continue
                
                # Get current price at this timestamp
                bars_so_far = bars.loc[:ts]
                if len(bars_so_far) == 0:
                    continue
                
                curr_price = float(bars_so_far['close'].iloc[-1])
                
                try:
                    current_tsl = float(bars_so_far['mtf_tsl'].iloc[-1])
                except (KeyError, IndexError):
                    current_tsl = None
                
                is_closed, close_type, pnl = manage_position(pos, curr_price, ts, params, current_mtf_tsl=current_tsl)
                
                if is_closed:
                    side_str = pos['side'].upper()
                    action = "STOP" if close_type == 'stop' else "TARGET"
                    log(f"  [{day_str}] {action} {pos['ticker']} ({side_str}) @ {curr_price:.2f} | PnL: {pnl:+.2f}%")
                    
                    if close_type == 'stop':
                        daily_stopped_tickers.add(pos['ticker'])
                    
                    # BUG FIX / REALISTIC SIZING:
                    # Calculate dollar PnL based on the actual dollar amount deployed.
                    # PnL % represents the un-leveraged move of the underlying asset.
                    profit_loss_dollars = pos['position_dollars'] * (pnl / 100.0)
                    equity += profit_loss_dollars

                    trade_history.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': curr_price,
                        'pnl': pnl,
                        'type': close_type,
                        'regime': regime
                    })
                    positions.remove(pos)
            
            # 2. Scan for new entries (if under position limit)
            if len(positions) >= params.get('max_positions', 5):
                continue
                
            # Prevent entries outside regular market hours (after 15:50 or before 9:30)
            if ts.hour >= 16 or (ts.hour == 15 and ts.minute >= 50) or ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
                continue
            
            held_tickers = {p['ticker'] for p in positions}
            
            # Determine candidate tickers based on regime
            effective_regime = 'bull' if always_long else regime
            
            if effective_regime == 'bull':
                candidates = [p['ticker'] for p in top_picks if p['ticker'] not in held_tickers]
            else:
                # Bear: try shorting top picks + longing inverse ETFs
                candidates = [p['ticker'] for p in top_picks if p['ticker'] not in held_tickers]
                inv_candidates = [etf for etf in INVERSE_ETFS[:5] 
                                  if etf not in held_tickers and etf in intraday_data]
                candidates.extend(inv_candidates)
            
            valid_signals = []
            
            for ticker in candidates:
                if len(positions) >= params.get('max_positions', 5):
                    break
                if ticker in held_tickers:
                    continue
                if ticker in daily_stopped_tickers: # Anti-revenge logic
                    continue
                if ticker not in intraday_data:
                    continue
                
                bars = intraday_data[ticker]
                if bars is None or len(bars) == 0:
                    continue
                
                # Compute indicators on data UP TO this timestamp (no future bias)
                bars_so_far = bars.loc[:ts]
                if len(bars_so_far) < 16:  # production stack needs 16+ bars
                    continue
                
                try:
                    df_precomp = precompute_quant_metrics_for_ticker(bars_so_far)
                    df_precomp = precompute_mtf_tsl_for_ticker(df_precomp)
                    quant = build_quant_metrics(df_precomp)
                    quant['current_price'] = float(bars_so_far['close'].iloc[-1])
                except Exception as e:
                    import traceback
                    print(f"Error computing indicators for {ticker} at {ts}: {e}")
                    traceback.print_exc()
                    continue
                
                entry_price = quant['current_price']
                if entry_price <= 0:
                    continue
                
                atr = quant.get("atr_5m", 0.0)
                
                # Apply ATR fallback as per mock_technical_analyst
                if atr == 0.0:
                    price = quant.get("current_price", 0.0)
                    atr = price * 0.004 if price > 0 else 1.0
                    quant["atr_5m"] = atr 
                
                trend_score = daily_trend_scores.get(ticker, 20.0)
                is_inverse_etf = ticker in INVERSE_ETFS
                
                entered = False
                side = 'long'
                reason = ''
                
                if effective_regime == 'bull':
                    entered, reason = evaluate_bull_entry(quant, trend_score, sparm)
                    side = 'long'
                elif effective_regime == 'bear':
                    if is_inverse_etf:
                        entered, reason = evaluate_inverse_etf_entry(quant, trend_score, sparm)
                        side = 'long'
                    else:
                        entered, reason = evaluate_bear_entry(quant, trend_score, sparm)
                        side = 'short'

                if entered:
                    # Parse out hybrid score from reason string
                    import re
                    h_score = 0
                    m = re.search(r'Score=(\d+)', reason)
                    if m: h_score = int(m.group(1))
                    
                    valid_signals.append({
                        'ticker': ticker,
                        'side': side,
                        'quant': quant,
                        'entry_price': entry_price,
                        'atr': atr,
                        'h_score': h_score,
                        'reason': reason
                    })
                    
            # Process signals: Sort by Alpha (h_score) descending, take top 2
            valid_signals.sort(key=lambda x: x['h_score'], reverse=True)
            top_signals = valid_signals[:2]
            
            for sig in top_signals:
                if len(positions) >= params.get('max_positions', 5):
                    break
                    
                ticker = sig['ticker']
                side = sig['side']
                entry_price = sig['entry_price']
                atr = sig['atr']
                reason = sig['reason']
                
                # Verify Sector Cap
                ticker_sector = SP500_SECTORS.get(ticker, "Unknown")
                active_sectors = [SP500_SECTORS.get(p['ticker'], "Unknown") for p in positions]
                if ticker_sector in active_sectors and ticker_sector != "Unknown" and ticker not in INVERSE_ETFS:
                    log(f"  [{day_str}] {side.upper()} {ticker} REJECTED | Reason: Sector Cap ({ticker_sector})")
                    continue
                    
                # Volatility-Parity sizing + Margin Cap logic
                stop_r_val = params.get('stop_r', 3.0)
                target_r_val = params.get('target_r', 3.0)
                
                # ---------------------------------------------------------
                # DYNAMIC HALF-KELLY CRITERION SIZING
                # ---------------------------------------------------------
                stop_dist = stop_r_val * atr  # Assumes 'atr' is calculated from the current 5m bar

                kelly_fraction = 0.01 # Default to 1% risk while warming up

                if len(trade_history) >= 15:
                    recent_trades = trade_history[-50:]
                    wins = [t.get('pnl_pct', t.get('pnl', 0)) for t in recent_trades if t.get('pnl_pct', t.get('pnl', 0)) > 0]
                    losses = [abs(t.get('pnl_pct', t.get('pnl', 0))) for t in recent_trades if t.get('pnl_pct', t.get('pnl', 0)) <= 0]
                    
                    W = len(wins) / len(recent_trades)
                    avg_win = sum(wins) / len(wins) if wins else 0.01
                    avg_loss = sum(losses) / len(losses) if losses else 0.01
                    R = avg_win / avg_loss
                    
                    full_kelly = W - ((1 - W) / R)
                    full_kelly = max(0.005, full_kelly) # Floor at 0.5%
                    
                    kelly_fraction = full_kelly / 2.0 # Standard Half-Kelly
                    kelly_fraction = min(kelly_fraction, 0.04) # Hard cap at 4% risk per trade

                risk_dollars = equity * kelly_fraction

                # Convert Risk to Position Size
                ideal_qty = risk_dollars / stop_dist if stop_dist > 0 else 1
                ideal_position_dollars = ideal_qty * entry_price

                # Apply Time-Based Margin Caps (Portfolio Defense)
                current_deployed = sum(p['position_dollars'] for p in positions)
                is_morning = (ts.hour < 11)
                total_margin_limit = equity * 2.0 if is_morning else equity * 4.0
                available_margin = total_margin_limit - current_deployed

                if available_margin < 100:
                    log(f"  [{day_str}] {side.upper()} {ticker} REJECTED | Reason: Margined Out (MorningCap={is_morning})")
                    continue
                    
                position_dollars = min(ideal_position_dollars, available_margin)
                effective_leverage = position_dollars / equity
                
                if side == 'long':
                    stop = entry_price - stop_dist
                    target = entry_price + target_r_val * atr
                else:
                    stop = entry_price + stop_dist
                    target = entry_price - target_r_val * atr
                
                pos = {
                    'ticker': ticker,
                    'side': side,
                    'entry_price': entry_price,
                    'position_dollars': position_dollars,
                    'stop': stop,
                    'target': target,
                    'atr_5m': atr,
                    'leverage': effective_leverage,
                    'entry_time': ts,
                    'regime': regime,
                    'days_held': 0
                }
                positions.append(pos)
                held_tickers.add(ticker)
                
                log(f"  [{day_str}] {side.upper()} {ticker} @ {entry_price:.2f} | {reason} | Size: ${position_dollars:,.0f} | Lev: {effective_leverage:.1f}x | Regime: {regime.upper()}")
        
        # End of day: Check if we need to force close (intraday mode) OR increment hold days
        last_ts = execution_times[-1] if execution_times else None
        
        for pos in list(positions):
            ticker = pos['ticker']
            
            pnl_computed = False
            last_price = pos['entry_price']
            
            if ticker in intraday_data and intraday_data[ticker] is not None and len(intraday_data[ticker]) > 0:
                last_price = float(intraday_data[ticker]['close'].loc[:last_ts].iloc[-1]) if last_ts else float(intraday_data[ticker]['close'].iloc[-1])
            
            is_long = pos['side'] == 'long'
            
            if params.get('max_hold_days', 0) == 0:
                # Force close End of Day (Intraday Mode)
                if is_long:
                    pnl = ((last_price - pos['entry_price']) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
                else:
                    pnl = ((pos['entry_price'] - last_price) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
                
                # BUG FIX: scale equity change by risk_per_trade, not 100% of capital
                risk_pct = params.get('risk_per_trade', 0.05)
                equity_change_pct = (pnl / 100) * risk_pct
                equity *= (1 + equity_change_pct)
                trade_history.append({
                    'date': trade_date,
                    'ticker': ticker,
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': last_price,
                    'pnl': pnl,
                    'type': 'eod_close',
                    'regime': pos['regime']
                })
                log(f"  [{day_str}] EOD CLOSE {ticker} ({pos['side'].upper()}) @ {last_price:.2f} | PnL: {pnl:+.2f}%")
                positions.remove(pos)
            else:
                # Multi-day hold mode
                pos['days_held'] += 1
                if pos['days_held'] >= params.get('max_hold_days', 5):
                    if is_long:
                        pnl = ((last_price - pos['entry_price']) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
                    else:
                        pnl = ((pos['entry_price'] - last_price) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
                    
                    # BUG FIX: scale equity change by risk_per_trade
                    risk_pct = params.get('risk_per_trade', 0.05)
                    equity_change_pct = (pnl / 100) * risk_pct
                    equity *= (1 + equity_change_pct)
                    trade_history.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': last_price,
                        'pnl': pnl,
                        'type': 'time_stop',
                        'regime': pos['regime']
                    })
                    log(f"  [{day_str}] TIME STOP {ticker} ({pos['side'].upper()}) @ {last_price:.2f} | PnL: {pnl:+.2f}%")
                    positions.remove(pos)
        
        equity_curve.append({'date': trade_date, 'equity': equity})
        
    # After simulation loop, close any lingering positions
    for pos in list(positions):
        ticker = pos['ticker']
        last_price = pos['entry_price']
        if ticker in all_tickers:
            tdf = all_tickers[ticker]
            daily_slice = tdf.loc[:end_date]
            if not daily_slice.empty:
                last_price = float(daily_slice['close'].iloc[-1])
            
        is_long = pos['side'] == 'long'
        if is_long:
            pnl = ((last_price - pos['entry_price']) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
        else:
            pnl = ((pos['entry_price'] - last_price) / pos['entry_price'] * pos.get('leverage', 1.0)) * 100
        
        # BUG FIX: scale equity change by risk_per_trade
        risk_pct = params.get('risk_per_trade', 0.05)
        equity_change_pct = (pnl / 100) * risk_pct
        equity *= (1 + equity_change_pct)
        
        trade_history.append({
            'date': end_date,
            'ticker': ticker,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': last_price,
            'pnl': pnl,
            'type': 'eod_close',
            'regime': pos['regime']
        })
        log(f"  [FINAL] EOD CLOSE {ticker} ({pos['side'].upper()}) @ {last_price:.2f} | PnL: {pnl:+.2f}%")
    positions.clear()
    equity_curve.append({'date': end_date, 'equity': equity})
    
    # ── Performance Summary ──
    log("\n" + "=" * 70)
    log("BACKTEST RESULTS")
    log("=" * 70)
    
    total_trades = len(trade_history)
    if total_trades == 0:
        log("No trades executed.")
        return {'total_trades': 0}
    
    wins = [t for t in trade_history if t['pnl'] > 0]
    losses = [t for t in trade_history if t['pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades * 100
    
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = sum(abs(t['pnl']) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0
    avg_pnl = sum(t['pnl'] for t in trade_history) / total_trades
    
    # Regime breakdown
    bull_trades = [t for t in trade_history if t.get('regime') == 'bull']
    bear_trades = [t for t in trade_history if t.get('regime') == 'bear']
    
    bull_win_rate = len([t for t in bull_trades if t['pnl'] > 0]) / len(bull_trades) * 100 if bull_trades else 0
    bear_win_rate = len([t for t in bear_trades if t['pnl'] > 0]) / len(bear_trades) * 100 if bear_trades else 0
    
    # Max drawdown from equity curve
    eq_series = pd.Series([e['equity'] for e in equity_curve])
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max * 100
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0
    
    total_return = (equity - 10000) / 10000 * 100
    
    log(f"\nTotal Trades: {total_trades}")
    log(f"Wins: {win_count} | Losses: {loss_count} | Win Rate: {win_rate:.2f}%")
    log(f"Avg Return Per Trade: {avg_pnl:.4f}%")
    log(f"Profit Factor: {profit_factor:.2f}")
    log(f"Gross Profit: {gross_profit:.2f} | Gross Loss: {gross_loss:.2f}")
    
    log(f"\nBreakdown:")
    log(f"  Bull Regime: {len(bull_trades)} trades | Win Rate: {bull_win_rate:.1f}%")
    log(f"  Bear Regime: {len(bear_trades)} trades | Win Rate: {bear_win_rate:.1f}%")
    
    log(f"\nAccount Simulation (Simple Compounding):")
    log(f"Final Equity: ${equity:,.2f} | Total Return: {total_return:+.2f}%")
    log(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Optional print line formatting precisely matching V1 optimizer output
    # timestamp,strategy,profit_factor,win_rate,total_trades,avg_pnl,bull_pnl,bear_pnl,chop_pnl,min_regime_pnl,gross_profit,gross_loss
    bull_avg = sum(t['pnl'] for t in bull_trades) / len(bull_trades) if bull_trades else 0
    bear_avg = sum(t['pnl'] for t in bear_trades) / len(bear_trades) if bear_trades else 0
    
    log("\n[V1 CSV MATCH FORMAT]")
    log(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')},MTF Strict Execution (Momentum Breakout),{profit_factor:.2f},{win_rate:.2f},{total_trades},{avg_pnl:.4f},{bull_avg:.4f},{bear_avg:.4f},0.0000,0.0000,{gross_profit:.2f},{gross_loss:.2f}")
    
    # Save log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\nResults saved to {os.path.basename(LOG_FILE)}")
    
    return {
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_equity': equity,
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_win_rate': bull_win_rate,
        'bear_win_rate': bear_win_rate,
        'trade_history': trade_history,
        'equity_curve': equity_curve
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V2 Backtest Engine: 3-Layer Architecture")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--long-only", action="store_true", help="Force MTF Strict Long strategy only (ignore regime)")
    parser.add_argument("--no-fetch", action="store_true", help="Skip Alpaca 5m fetch (use cached only)")
    parser.add_argument("--param-file", type=str, default=None, help="Path to strategy parameters file (CSV 2-lines format)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    sparm = load_strategy_params(args.param_file)
    
    custom_params = {
        'stop_r': sparm['stop_r'],
        'target_r': sparm['target_r'],
        'risk_per_trade': 0.05,
        'max_positions': 5,
        'max_hold_days': 0,
    }
    
    results = run_backtest_v2(
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        fetch_5m=not args.no_fetch,
        verbose=not args.quiet,
        always_long=args.long_only,
        params=custom_params,
        sparm=sparm
    )
