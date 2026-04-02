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


def _et_hour_minute(ts) -> tuple[int, int]:
    """
    Return (hour, minute) in US/Eastern time.
    Handles both UTC-aware timestamps (from Alpaca parquet) and naive timestamps.
    The Alpaca SDK returns UTC-aware datetimes; if stored with timezone info intact,
    ts.hour would be UTC (e.g. 14 for 9:30 AM ET), breaking naive hour comparisons.
    """
    if getattr(ts, 'tzinfo', None) is not None:
        try:
            import pytz
            ts = ts.astimezone(pytz.timezone('America/New_York'))
        except Exception:
            pass  # Fall back to raw hour if pytz unavailable
    return ts.hour, ts.minute


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def load_strategy_params(filepath: str) -> dict:
    """
    Load dynamic strategy parameters from a CSV-like txt file.
    Format expected:
    mom_vwap_z_min,mom_vol_ratio_min,mom_delta_min,rev_vwap_z_max,rev_vol_spike_min,stop_r,target_r,risk_per_trade
    Input: 0.3,1.3,0.58,-2.5,2.0,1.5,2.0,0.05
    """
    default_params = {
        'mom_vwap_z_min': 0.3,
        'mom_vol_ratio_min': 1.3,
        'mom_delta_min': 0.58,
        'rev_vwap_z_max': -2.5,
        'rev_vol_spike_min': 2.0,
        'stop_r': 1.5,
        'target_r': 2.0,
        'risk_per_trade': 0.05,
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


def evaluate_momentum_entry(quant: dict, trend_score: float, sparm: dict) -> tuple[bool, str]:
    """
    True Momentum: VWAP breakout with volume expansion and buy-side dominance.
    Requires price ABOVE VWAP (trend continuation), not below.

    Conditions (all must be true):
      1. vwap_zscore >= mom_vwap_z_min  (price above VWAP)
      2. vwap_zscore <= 2.5             (not extremely overbought)
      3. volume_ratio >= mom_vol_ratio_min  (volume expanding vs 20-bar SMA)
      4. delta_ratio >= mom_delta_min   (buy-side dominance)
      5. trend_score >= 22.4            (daily trend alignment; 56/100 * 0.40 on 0-40 scale)
    """
    z = quant.get('vwap_zscore', 0)
    vol_ratio = quant.get('volume_ratio', 1.0)
    dr = quant.get('delta_ratio', 0.5)

    # trend_score is on 0-40 scale. 56 on the 0-100 raw scale = 22.4 here.
    min_trend = 22.4

    entry = (
        z >= sparm['mom_vwap_z_min'] and           # price above VWAP
        z <= 2.5 and                                # not extremely overbought
        vol_ratio >= sparm['mom_vol_ratio_min'] and # volume expanding vs 20-bar SMA
        dr >= sparm['mom_delta_min'] and            # buy-side dominance
        trend_score >= min_trend                     # daily trend alignment
    )

    reason = f"MOM Z={z:.2f} VR={vol_ratio:.1f} DR={dr:.2f} TS={trend_score:.1f}"
    return entry, reason


def evaluate_mean_reversion_entry(quant: dict, trend_score: float, sparm: dict) -> tuple[bool, str]:
    """
    Confirmed Capitulation + Reversal Bar.
    Requires deep VWAP extension, RSI oversold, volume SPIKE, and smart_bounce.

    Conditions (all must be true):
      1. vwap_zscore <= rev_vwap_z_max  (deep below VWAP)
      2. rsi_14 < 32                    (genuinely oversold)
      3. volume_ratio >= rev_vol_spike_min  (capitulation volume spike)
      4. smart_bounce == True           (reversal bar: buy_frac > 0.75 AND vol > 1.5x SMA)
      5. delta_ratio <= 0.38            (selling pressure over 20 bars)
      6. trend_score >= 8.0             (not in total structural freefall; 20/100 * 0.40)
    """
    z = quant.get('vwap_zscore', 0)
    rsi = quant.get('rsi_14', 50.0)
    vol_ratio = quant.get('volume_ratio', 1.0)
    smart_bounce = quant.get('smart_bounce', False)
    dr = quant.get('delta_ratio', 0.5)

    # trend_score floor: 20 on 0-100 scale = 8.0 on 0-40 scale
    min_trend = 8.0

    entry = (
        z <= sparm['rev_vwap_z_max'] and              # deep below VWAP
        rsi < 32.0 and                                 # genuinely oversold
        vol_ratio >= sparm['rev_vol_spike_min'] and    # capitulation volume spike
        smart_bounce and                                # reversal bar: buyers stepping in
        dr <= 0.38 and                                  # selling pressure over 20 bars
        trend_score >= min_trend                        # not in total structural freefall
    )

    reason = f"REV Z={z:.2f} RSI={rsi:.1f} VR={vol_ratio:.1f} SB={smart_bounce} DR={dr:.2f}"
    return entry, reason


# ──────────────────────────────────────────────────────────────────────────────
# INTRADAY REGIME DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _intraday_regime(spy_bars_so_far) -> str:
    """
    Fast intraday regime: SPY EMA(20) check on 5-minute bars.
    Used to scale position size, not to hard-switch strategies.
    - Bull: SPY above 20-bar EMA
    - Bear: SPY below 20-bar EMA AND 10-bar ROC < -0.3%
    - Chop: everything else
    """
    if spy_bars_so_far is None or len(spy_bars_so_far) < 20:
        return 'chop'
    spy_close = float(spy_bars_so_far['close'].iloc[-1])
    spy_ema20 = float(spy_bars_so_far['close'].ewm(span=20, adjust=False).mean().iloc[-1])
    if spy_close > spy_ema20:
        return 'bull'
    if len(spy_bars_so_far) >= 10:
        spy_roc = spy_close / float(spy_bars_so_far['close'].iloc[-10]) - 1.0
        if spy_roc < -0.003:
            return 'bear'
    return 'chop'


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
    
    _ct_hour, _ct_min = _et_hour_minute(current_time)
    is_morning_lock = _ct_hour == 9 or (_ct_hour == 10 and _ct_min <= 30)
    circuit_breaker_loss_limit = -2.0 * leverage  # Scale with leverage, not fixed -4%
    
    dir_mult = 1.0 if is_long else -1.0
    open_pnl = ((current_price - entry_price) / entry_price * dir_mult) * 100 * leverage
    
    if is_morning_lock:
        if open_pnl <= circuit_breaker_loss_limit:
            return True, 'stop', open_pnl
        elif current_price >= pos['target'] if is_long else current_price <= pos['target']:
            return True, 'target', open_pnl
        return False, '', 0.0

    # --- DYNAMIC MTF TSL LOGIC ---
    # Only engage trailing stop after trade has moved >= 1× ATR in profit.
    # Without this gate, the 5m-based TSL (2× ATR_5m ≈ 0.6%) immediately
    # overrides the wider daily-ATR stop (≈ 0.75-0.9%), causing constant
    # same-bar stop-outs with tiny losses (30% WR → negative EV).
    trail_threshold = atr  # 1× ATR_5m in the right direction
    if is_long:
        unrealized_move = current_price - entry_price
    else:
        unrealized_move = entry_price - current_price
    trade_in_profit = unrealized_move >= trail_threshold

    if trade_in_profit and current_mtf_tsl is not None and not np.isnan(current_mtf_tsl):
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
    sparm: dict = None,
    retrain_interval_days: int = 21,
    regime_dwell_days: int = 3,
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
            'stop_r': sparm.get('stop_r', 1.5),
            'target_r': sparm.get('target_r', 2.0),
            'risk_per_trade': 0.05,          # base risk fraction per trade
            'target_trade_vol': 2.0,         # target std of per-trade returns (%) for heat scaling
            'max_total_leverage': 1.5,       # total deployed capital / equity cap
            'max_positions': 5,
            'max_hold_trading_days': 10,     # EOD force-close after N trading days (0 = off)
            'ticker_cooldown_days': 3,       # days a stopped ticker is blacklisted after normal stop
            'loss_cooldown_threshold': -1.5, # % PnL below which extended cooldown applies
            'ticker_cooldown_extended_days': 10,  # extended blacklist days for large losses
            'slippage_pct': 0.0003,          # 0.03% per leg (bid-ask half-spread for liquid stocks)
            'commission_pct': 0.0,           # Alpaca: $0 commissions; SEC/FINRA fees negligible (<0.003%)
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
        verbose=verbose,
        retrain_interval_days=retrain_interval_days,
        regime_dwell_days=regime_dwell_days,
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
    ticker_cooldown: dict = {}  # ticker → day_idx when cooldown expires (after stop-outs)

    for day_idx, snapshot in enumerate(daily_snapshots):
        trade_date = snapshot['date']
        regime = snapshot['regime']
        top_picks = snapshot['top_tickers']
        intraday_data = snapshot.get('intraday_data', {})
        daily_stopped_tickers = set()   # Tickers stopped out today — no re-entry
        daily_targeted_tickers = set()  # Tickers that hit target today — cool off, no re-entry
        
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
        daily_atr_pcts = {}  # Daily ATR as fraction of price — used for position sizing & stop floor
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
                    # Compute daily ATR for position sizing (reuse existing compute_atr)
                    if highs_list and lows_list and len(closes_list) >= 15:
                        from utils.indicators import compute_atr
                        daily_atr_val = compute_atr(highs_list, lows_list, closes_list, period=14)
                        if daily_atr_val > 0 and closes_list[-1] > 0:
                            daily_atr_pcts[ticker] = max(daily_atr_val / closes_list[-1], 0.005)
                else:
                    daily_trend_scores[ticker] = 20.0

        # ── SPY Market Context: compute once per day, no look-ahead ──
        # spy_trending_down: blocks dip-buy entries in sustained downtrends.
        # spy_trending_up:   blocks inverse ETF entries in sustained uptrends.
        #   Symmetric guards — dip-buys need a rising context; inverse ETFs need a falling one.
        spy_trending_down = False  # default: allow dip-buy
        spy_trending_up   = False  # default: allow inverse ETF
        SPY_TICKER = 'SPY'
        if SPY_TICKER in all_tickers:
            spy_slice_end = trade_date - pd.Timedelta(days=1)
            spy_daily = all_tickers[SPY_TICKER].loc[:spy_slice_end]['close']
            if len(spy_daily) >= 22:
                spy_ma20  = spy_daily.iloc[-20:].mean()
                spy_close = spy_daily.iloc[-1]
                spy_roc10 = (spy_close / spy_daily.iloc[-11] - 1.0) if len(spy_daily) >= 12 else 0.0
                spy_trending_down = (spy_close < spy_ma20) and (spy_roc10 < -0.01)
                spy_trending_up   = (spy_close > spy_ma20) and (spy_roc10 >  0.01)

        # ── Daily EMA alignment for dip-buy quality filter ──────────────────
        # Only buy dips on stocks that are in a daily uptrend (EMA20 > EMA50).
        # If the stock has been downtrending for weeks, an intraday dip is a
        # continuation, not a mean-reversion opportunity.
        daily_ema_uptrend = {}
        for pick in top_picks:
            tkr = pick['ticker']
            if tkr in all_tickers:
                tdf_daily = all_tickers[tkr]
                slice_end = trade_date - pd.Timedelta(days=1)
                closes_daily = tdf_daily.loc[:slice_end]['close']
                if len(closes_daily) >= 50:
                    ema20 = float(closes_daily.ewm(span=20, adjust=False).mean().iloc[-1])
                    ema50 = float(closes_daily.ewm(span=50, adjust=False).mean().iloc[-1])
                    daily_ema_uptrend[tkr] = ema20 > ema50
                else:
                    daily_ema_uptrend[tkr] = True  # insufficient history → assume uptrend

        # ── Precompute full-day quant metrics (performance) ──────────────────
        # precompute_quant_metrics_for_ticker + precompute_mtf_tsl_for_ticker are
        # O(n) but called per-bar × per-candidate = O(n²) total.
        # Running them ONCE per day per ticker then slicing is O(n), ~10× faster.
        # No look-ahead: all indicators are causal (EWM/cumsum from bar 0 forward).
        all_eval_tickers = (
            {p['ticker'] for p in top_picks}
            | {p['ticker'] for p in positions}
            | ({SPY_TICKER})
        )
        if regime in ('bear', 'chop'):
            all_eval_tickers.update(INVERSE_ETFS[:5])
        ticker_day_metrics: dict = {}
        for eval_tkr in all_eval_tickers:
            if eval_tkr not in intraday_data or intraday_data[eval_tkr] is None:
                continue
            full_bars = intraday_data[eval_tkr]
            if len(full_bars) == 0:
                continue
            try:
                df_p = precompute_quant_metrics_for_ticker(full_bars)
                df_p = precompute_mtf_tsl_for_ticker(df_p)
                ticker_day_metrics[eval_tkr] = df_p
            except Exception:
                pass

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

                # Use precomputed TSL (column is 'tsl_1', not 'mtf_tsl')
                current_tsl = None
                if ticker in ticker_day_metrics:
                    try:
                        dm_at_ts = ticker_day_metrics[ticker].loc[:ts]
                        if len(dm_at_ts) > 0 and 'tsl_1' in dm_at_ts.columns:
                            current_tsl = float(dm_at_ts['tsl_1'].iloc[-1])
                    except Exception:
                        pass

                # Mean reversion exits: VWAP reversion target + 30-bar time stop
                if pos.get('strategy_type') == 'mean_reversion':
                    # VWAP reversion target: exit when price returns near VWAP
                    if ticker in ticker_day_metrics:
                        try:
                            dm_slice = ticker_day_metrics[ticker].loc[:ts]
                            if len(dm_slice) > 0 and 'vwap_zscore' in dm_slice.columns:
                                current_z = float(dm_slice['vwap_zscore'].iloc[-1])
                                if current_z >= -0.5:
                                    is_long  = pos['side'] == 'long'
                                    lev      = pos.get('leverage', 1.0)
                                    slip_pct = params.get('slippage_pct', 0.0)
                                    comm_pct = params.get('commission_pct', 0.0)
                                    eff_exit_rv = curr_price * (1.0 - slip_pct) if is_long else curr_price * (1.0 + slip_pct)
                                    pnl_rv = ((eff_exit_rv - pos['entry_price']) / pos['entry_price'] * (1 if is_long else -1)) * 100 * lev
                                    entry_equity_rv = pos['position_dollars'] / lev
                                    equity += (pnl_rv / 100.0) * entry_equity_rv
                                    if comm_pct > 0.0:
                                        equity -= entry_equity_rv * comm_pct
                                    log(f"  [{day_str}] VWAP-REVERT {ticker} @ {curr_price:.2f} | PnL: {pnl_rv:+.2f}%")
                                    trade_history.append({'date': trade_date, 'ticker': ticker, 'side': pos['side'],
                                                          'entry_price': pos['entry_price'], 'exit_price': curr_price,
                                                          'pnl': pnl_rv, 'profit_dollars': (pnl_rv / 100.0) * entry_equity_rv,
                                                          'type': 'vwap_revert', 'regime': regime,
                                                          'strategy_type': 'mean_reversion'})
                                    positions.remove(pos)
                                    continue
                        except Exception:
                            pass

                    # 30-bar time stop (2.5 hours): thesis invalidated if no snap-back
                    try:
                        bars_held = int((ts - pos['entry_time']).total_seconds() / 300)
                        if bars_held >= 30:
                            is_long  = pos['side'] == 'long'
                            lev      = pos.get('leverage', 1.0)
                            slip_pct = params.get('slippage_pct', 0.0)
                            comm_pct = params.get('commission_pct', 0.0)
                            eff_exit_tb = curr_price * (1.0 - slip_pct) if is_long else curr_price * (1.0 + slip_pct)
                            pnl_tb = ((eff_exit_tb - pos['entry_price']) / pos['entry_price'] * (1 if is_long else -1)) * 100 * lev
                            entry_equity_tb = pos['position_dollars'] / lev
                            equity += (pnl_tb / 100.0) * entry_equity_tb
                            if comm_pct > 0.0:
                                equity -= entry_equity_tb * comm_pct
                            log(f"  [{day_str}] TIMEOUT {ticker} (MEAN-REV) @ {curr_price:.2f} | PnL: {pnl_tb:+.2f}% | Held {bars_held} bars")
                            trade_history.append({'date': trade_date, 'ticker': ticker, 'side': pos['side'],
                                                  'entry_price': pos['entry_price'], 'exit_price': curr_price,
                                                  'pnl': pnl_tb, 'profit_dollars': (pnl_tb / 100.0) * entry_equity_tb,
                                                  'type': 'timeout', 'regime': regime,
                                                  'strategy_type': 'mean_reversion'})
                            positions.remove(pos)
                            continue
                    except Exception:
                        pass

                is_closed, close_type, pnl = manage_position(pos, curr_price, ts, params, current_mtf_tsl=current_tsl)

                if is_closed:
                    # Apply exit-leg slippage: selling long at bid (below bar close), covering short at ask
                    slip_pct = params.get('slippage_pct', 0.0)
                    comm_pct = params.get('commission_pct', 0.0)
                    is_long  = pos['side'] == 'long'
                    lev      = pos.get('leverage', 1.0)
                    eff_exit = curr_price * (1.0 - slip_pct) if is_long else curr_price * (1.0 + slip_pct)
                    # Recompute pnl using slippage-adjusted exit price
                    if is_long:
                        pnl = ((eff_exit - pos['entry_price']) / pos['entry_price'] * lev) * 100
                    else:
                        pnl = ((pos['entry_price'] - eff_exit) / pos['entry_price'] * lev) * 100
                    # Deduct exit-leg commission from equity
                    if comm_pct > 0.0:
                        equity -= (pos['position_dollars'] / lev) * comm_pct

                    side_str = pos['side'].upper()
                    action = "STOP" if close_type == 'stop' else "TARGET"
                    log(f"  [{day_str}] {action} {pos['ticker']} ({side_str}) @ {curr_price:.2f} | PnL: {pnl:+.2f}%")

                    if close_type == 'stop':
                        daily_stopped_tickers.add(pos['ticker'])
                        cooldown_days = params.get('ticker_cooldown_days', 3)
                        # Loss-triggered extended cooldown: large stop-outs get a longer blacklist
                        # (e.g. -8% CVNA-style wipeout should not be allowed back for weeks)
                        loss_threshold = params.get('loss_cooldown_threshold', -1.5)
                        extended_days  = params.get('ticker_cooldown_extended_days', 10)
                        if pnl < loss_threshold:
                            applied_cooldown = extended_days
                        else:
                            applied_cooldown = cooldown_days
                        if applied_cooldown > 0:
                            ticker_cooldown[pos['ticker']] = day_idx + applied_cooldown
                    elif close_type == 'target':
                        daily_targeted_tickers.add(pos['ticker'])  # Cool off after target hit
                    
                    # Convert the account-relative % PnL into raw dollars using the entry equity.
                    # This prevents double-compounding math errors on overlapping concurrent trades.
                    entry_equity = pos['position_dollars'] / pos.get('leverage', 1.0)
                    profit_dollars = (pnl / 100.0) * entry_equity
                    equity += profit_dollars

                    trade_history.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': curr_price,
                        'pnl': pnl,
                        'profit_dollars': profit_dollars,
                        'type': close_type,
                        'regime': regime,
                        'strategy_type': pos.get('strategy_type', 'momentum'),
                    })
                    positions.remove(pos)
            
            # 2. Scan for new entries (if under position limit)
            if equity <= 0:
                log(f"  [{day_str}] BANKRUPT — equity ${equity:,.2f}, halting new entries")
                continue
            if len(positions) >= params.get('max_positions', 5):
                continue
                
            # Prevent entries outside regular market hours (after 15:50 or before 9:30 ET)
            # Uses ET conversion so the filter is correct for both UTC-aware (Alpaca) and naive timestamps.
            _ts_hour, _ts_min = _et_hour_minute(ts)
            if _ts_hour >= 16 or (_ts_hour == 15 and _ts_min >= 50) or _ts_hour < 9 or (_ts_hour == 9 and _ts_min < 30):
                continue
            
            held_tickers = {p['ticker'] for p in positions}
            
            # ── Fast intraday regime (SPY EMA20) ─────────────────────────────
            spy_5m_so_far = None
            if SPY_TICKER in ticker_day_metrics:
                spy_5m_so_far = ticker_day_metrics[SPY_TICKER].loc[:ts]
            intraday_regime = _intraday_regime(spy_5m_so_far)

            # ── Helper: evaluate one ticker and return a signal dict or None ──
            def _eval_ticker(ticker, eval_fn, trend_scores):
                if ticker in held_tickers:            return None
                if ticker in daily_stopped_tickers:   return None
                if ticker in daily_targeted_tickers:  return None  # Already hit target today
                if ticker_cooldown.get(ticker, 0) > day_idx: return None  # Multi-day stop cooldown
                # Use precomputed daily metrics (O(1) per bar instead of O(n²))
                if ticker not in ticker_day_metrics:  return None
                bars_so_far = ticker_day_metrics[ticker].loc[:ts]
                if len(bars_so_far) < 16:             return None
                try:
                    quant = build_quant_metrics(bars_so_far)  # fast-path: reads from precomputed cols
                    quant['current_price'] = float(bars_so_far['close'].iloc[-1])
                except Exception as e:
                    return None
                entry_price = quant['current_price']
                if entry_price <= 0: return None
                atr = quant.get("atr_5m", 0.0)
                if atr == 0.0:
                    atr = entry_price * 0.004 if entry_price > 0 else 1.0
                    quant["atr_5m"] = atr
                trend_score = trend_scores.get(ticker, 20.0)
                entered, reason = eval_fn(quant, trend_score, sparm)
                if not entered: return None
                # No hybrid score in new strategy — fixed neutral conviction;
                # regime_multiplier handles scaling instead.
                h_score = 55
                return {'ticker': ticker, 'side': 'long', 'quant': quant,
                        'entry_price': entry_price, 'atr': atr,
                        'h_score': h_score, 'reason': reason,
                        'daily_atr_pct': daily_atr_pcts.get(ticker, 0.015)}

            valid_signals = []
            max_pos = params.get('max_positions', 5)
            candidates = [p['ticker'] for p in top_picks if p['ticker'] not in held_tickers]

            for ticker in candidates:
                if len(positions) + len(valid_signals) >= max_pos:
                    break

                # Momentum entry: only in bull regime
                if intraday_regime == 'bull':
                    sig = _eval_ticker(ticker, evaluate_momentum_entry, daily_trend_scores)
                    if sig:
                        sig['strategy_type'] = 'momentum'
                        sig['regime_multiplier'] = 1.0
                        valid_signals.append(sig)
                        continue

                # Mean reversion entry: all regimes, regime scales size
                sig = _eval_ticker(ticker, evaluate_mean_reversion_entry, daily_trend_scores)
                if sig:
                    sig['strategy_type'] = 'mean_reversion'
                    if intraday_regime == 'bull':
                        sig['regime_multiplier'] = 0.7
                    elif intraday_regime == 'chop':
                        sig['regime_multiplier'] = 1.0
                    elif intraday_regime == 'bear':
                        # Only extreme setups in bear
                        z_val = sig['quant'].get('vwap_zscore', 0)
                        vr_val = sig['quant'].get('volume_ratio', 1.0)
                        if z_val > -3.5 or vr_val < 3.0:
                            continue  # reject non-extreme setups in bear
                        sig['regime_multiplier'] = 0.4
                    valid_signals.append(sig)

            # Process signals: take top 2 overall
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
                
                # Volatility-Parity sizing + Margin Cap logic
                is_mean_rev = sig.get('strategy_type') == 'mean_reversion'
                stop_r_val = params.get('stop_r', 1.5)
                target_r_val = params.get('target_r', 2.0)

                # ---------------------------------------------------------
                # DYNAMIC VOLATILITY PARITY SIZING
                # ---------------------------------------------------------
                # Stop placement: use 5m ATR, but floor at 0.5× daily ATR to prevent
                # intraday noise from triggering stops on normal oscillation.
                # Mean reversion uses a fixed tight 1.5× stop (VWAP revert is the primary exit).
                effective_stop_r = 1.5 if is_mean_rev else stop_r_val
                stop_dist_5m = effective_stop_r * atr
                daily_atr_dollars = sig.get('daily_atr_pct', 0.015) * entry_price
                stop_dist = max(stop_dist_5m, 0.5 * daily_atr_dollars)

                base_risk_fraction = params.get('risk_per_trade', 0.05)

                # Conviction Scaling: disabled (h_score not yet dynamic in v2).
                # When a real conviction signal is available, map h_score
                # from (30 → 0.5) to (70 → 1.5) here.
                conviction_multiplier = 1.0

                # Regime Scaling: driven by intraday regime via signal dict
                regime_multiplier = sig.get('regime_multiplier', 1.0)

                # Portfolio Heat Scaling — adaptive, no hardcoded cap.
                # Measures recent trade-return volatility and scales size down when
                # the portfolio is running "hot" (e.g. a losing streak inflates vol).
                # min 5 trades for a meaningful sample; target_trade_vol is the desired
                # per-trade return std in % (e.g. 2.0 means we expect ~2% std per trade).
                if len(trade_history) >= 5:
                    recent_pnls = [t['pnl'] for t in trade_history[-20:]]
                    recent_vol = float(np.std(recent_pnls)) if len(recent_pnls) > 1 else 2.0
                    target_trade_vol = params.get('target_trade_vol', 2.0)
                    heat_scalar = min(1.2, target_trade_vol / max(recent_vol, 0.3))
                else:
                    heat_scalar = 1.0  # no history yet — full base size

                # Diversification Scaling — mathematically grounded, no hardcoded cap.
                # Portfolio variance with N correlated positions scales sub-linearly.
                # 0 open→1.0x, 1 open→0.71x, 2→0.58x, 3→0.50x, 4→0.45x
                n_open = len(positions)
                div_scalar = 1.0 / np.sqrt(n_open + 1)

                dynamic_risk_fraction = (
                    base_risk_fraction
                    * conviction_multiplier
                    * regime_multiplier
                    * heat_scalar
                    * div_scalar
                )
                risk_dollars = equity * dynamic_risk_fraction

                # Convert Risk to Position Size using daily ATR-scaled volatility parity.
                # The actual stop is at stop_r × ATR_5m (tight intraday stop), but
                # position SIZE uses the stock's daily ATR as a floor. This makes
                # risk_per_trade meaningful without the old arbitrary 4% floor that
                # amplified losses by over-sizing positions.
                stop_pct = stop_dist / entry_price
                daily_atr_pct = sig.get('daily_atr_pct', 0.015)
                if stop_pct > 0:
                    sizing_stop = max(stop_pct, daily_atr_pct)
                    ideal_position_dollars = risk_dollars / sizing_stop
                else:
                    ideal_position_dollars = equity * 0.2  # Fallback
                    
                # Portfolio-Level Leverage Cap (makes div_scalar actually effective)
                # Instead of capping each position independently at 1.0× equity,
                # we cap TOTAL deployed capital at max_total_leverage × equity.
                # With multiple open positions, available_capital shrinks naturally —
                # div_scalar reduces risk_dollars and the leverage cap reduces position
                # size through two independent, compounding channels.
                # Example (equity=$10K, max_total_leverage=1.5, 2 open @$7.5K each):
                #   available = $15K - $15K = $0 → new entries are REJECTED, not squeezed
                target_total_leverage = params.get('max_total_leverage', 1.5)
                current_deployed = sum(p['position_dollars'] for p in positions)
                available_capital = max(0.0, equity * target_total_leverage - current_deployed)

                if available_capital < 100:
                    continue  # Leverage cap reached — silent (expected behaviour, not an error)

                # Per-position leverage caps:
                #   Mean reversion: 0.5× equity (lower-conviction)
                #   Momentum: 1.0× equity (no single trade exceeds equity)
                max_per_trade = equity * 0.5 if is_mean_rev else equity * 1.0
                position_dollars = min(ideal_position_dollars, available_capital, max_per_trade)
                effective_leverage = position_dollars / equity

                # ── Slippage: adjust entry price to simulate bid-ask spread ──
                # Buying a long: we pay the ask (above mid). Shorting: we receive bid (below mid).
                slip_pct  = params.get('slippage_pct', 0.0)
                comm_pct  = params.get('commission_pct', 0.0)
                if side == 'long':
                    eff_entry = entry_price * (1.0 + slip_pct)
                else:
                    eff_entry = entry_price * (1.0 - slip_pct)

                # Mean reversion uses a wide ATR hard cap (VWAP revert exit fires first).
                # Momentum trades use the optimizer-tuned target_r.
                if is_mean_rev:
                    effective_target_r = target_r_val * 2  # hard cap; VWAP reversion fires first
                else:
                    effective_target_r = target_r_val

                if side == 'long':
                    stop = eff_entry - stop_dist
                    target = eff_entry + effective_target_r * atr
                else:
                    stop = eff_entry + stop_dist
                    target = eff_entry - effective_target_r * atr

                # Deduct entry-leg commission immediately
                if comm_pct > 0.0:
                    equity -= (position_dollars / effective_leverage) * comm_pct

                pos = {
                    'ticker': ticker,
                    'side': side,
                    'entry_price': eff_entry,
                    'raw_entry_price': entry_price,   # pre-slippage, for logging
                    'position_dollars': position_dollars,
                    'stop': stop,
                    'target': target,
                    'atr_5m': atr,
                    'leverage': effective_leverage,
                    'entry_time': ts,
                    'regime': regime,
                    'days_held': 0,
                    'strategy_type': sig.get('strategy_type', 'momentum'),
                }
                positions.append(pos)
                held_tickers.add(ticker)

                log(f"  [{day_str}] {side.upper()} {ticker} @ {entry_price:.2f} (eff={eff_entry:.2f}) | {reason} | Size: ${position_dollars:,.0f} | Lev: {effective_leverage:.1f}x | Regime: {intraday_regime.upper()}")
        
        # ── EOD: Increment hold-day counter, force-close stale momentum positions ──
        # Mean reversion positions have a 30-bar intraday timeout already.
        # Momentum entries are subject to max_hold_trading_days.
        max_hold = params.get('max_hold_trading_days', 0)
        for pos in list(positions):
            pos['days_held'] = pos.get('days_held', 0) + 1
            if max_hold > 0 and pos.get('strategy_type') != 'mean_reversion' and pos['days_held'] >= max_hold:
                ticker = pos['ticker']
                last_price = pos['entry_price']  # fallback: flat PnL if no intraday data
                if ticker in intraday_data and intraday_data[ticker] is not None and len(intraday_data[ticker]) > 0:
                    last_price = float(intraday_data[ticker]['close'].iloc[-1])
                is_long  = pos['side'] == 'long'
                lev      = pos.get('leverage', 1.0)
                slip_pct = params.get('slippage_pct', 0.0)
                comm_pct = params.get('commission_pct', 0.0)
                eff_exit_hold = last_price * (1.0 - slip_pct) if is_long else last_price * (1.0 + slip_pct)
                pnl_hold = ((eff_exit_hold - pos['entry_price']) / pos['entry_price'] * (1 if is_long else -1)) * 100 * lev
                entry_equity_hold = pos['position_dollars'] / lev
                profit_dollars_hold = (pnl_hold / 100.0) * entry_equity_hold
                equity += profit_dollars_hold
                if comm_pct > 0.0:
                    equity -= entry_equity_hold * comm_pct
                log(f"  [{day_str}] MAX-HOLD {ticker} ({pos['side'].upper()}) @ {last_price:.2f} | PnL: {pnl_hold:+.2f}% | Held {pos['days_held']} days")
                trade_history.append({
                    'date': trade_date, 'ticker': ticker, 'side': pos['side'],
                    'entry_price': pos['entry_price'], 'exit_price': last_price,
                    'pnl': pnl_hold, 'profit_dollars': profit_dollars_hold,
                    'type': 'max_hold', 'regime': regime,
                    'strategy_type': pos.get('strategy_type', 'momentum'),
                })
                positions.remove(pos)

        # Mark-to-market open positions at EOD close for accurate drawdown tracking.
        # Realized equity only changes on trade close, so without MTM the equity
        # curve misses intraday adverse moves on held positions.
        mtm_equity = equity
        for pos in positions:
            ticker = pos['ticker']
            if ticker in intraday_data and intraday_data[ticker] is not None and len(intraday_data[ticker]) > 0:
                mtm_price = float(intraday_data[ticker]['close'].iloc[-1])
                lev = pos.get('leverage', 1.0)
                entry_eq = pos['position_dollars'] / lev
                if pos['side'] == 'long':
                    unrealized = (mtm_price - pos['entry_price']) / pos['entry_price'] * lev * entry_eq
                else:
                    unrealized = (pos['entry_price'] - mtm_price) / pos['entry_price'] * lev * entry_eq
                mtm_equity += unrealized
        equity_curve.append({'date': trade_date, 'equity': mtm_equity})
        
    # After simulation loop, close any lingering positions at last known 5m price.
    # IMPORTANT: Use intraday 5m bar price (same source as entry price) NOT the daily
    # CSV which stores backward split-adjusted prices and will produce wildly wrong PnL.
    last_day_intraday = daily_snapshots[-1].get('intraday_data', {}) if daily_snapshots else {}
    for pos in list(positions):
        ticker = pos['ticker']
        last_price = pos['entry_price']  # Fallback: flat PnL if no closing price found
        if ticker in last_day_intraday and last_day_intraday[ticker] is not None and len(last_day_intraday[ticker]) > 0:
            last_price = float(last_day_intraday[ticker]['close'].iloc[-1])

        is_long  = pos['side'] == 'long'
        lev      = pos.get('leverage', 1.0)
        slip_pct = params.get('slippage_pct', 0.0)
        comm_pct = params.get('commission_pct', 0.0)
        eff_last = last_price * (1.0 - slip_pct) if is_long else last_price * (1.0 + slip_pct)
        if is_long:
            pnl = ((eff_last - pos['entry_price']) / pos['entry_price'] * lev) * 100
        else:
            pnl = ((pos['entry_price'] - eff_last) / pos['entry_price'] * lev) * 100

        # Convert the account-relative % PnL into raw dollars using the entry equity.
        # This prevents double-compounding math errors on overlapping concurrent trades.
        entry_equity = pos['position_dollars'] / lev
        profit_dollars = (pnl / 100.0) * entry_equity
        equity += profit_dollars
        if comm_pct > 0.0:
            equity -= entry_equity * comm_pct

        trade_history.append({
            'date': end_date,
            'ticker': ticker,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': last_price,
            'pnl': pnl,
            'profit_dollars': profit_dollars,
            'type': 'eod_close',
            'regime': pos['regime'],
            'strategy_type': pos.get('strategy_type', 'momentum'),
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
    
    gross_profit_dollars = sum(t['profit_dollars'] for t in wins)
    gross_loss_dollars = sum(abs(t['profit_dollars']) for t in losses)
    profit_factor = gross_profit_dollars / gross_loss_dollars if gross_loss_dollars > 0 else float('inf')
    
    avg_win = gross_profit_dollars / win_count if win_count > 0 else 0
    avg_loss = gross_loss_dollars / loss_count if loss_count > 0 else 0
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
    log(f"Gross Profit: ${gross_profit_dollars:,.2f} | Gross Loss: ${gross_loss_dollars:,.2f}")
    
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
    log(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')},MTF Strict Execution (Momentum Breakout),{profit_factor:.2f},{win_rate:.2f},{total_trades},{avg_pnl:.4f},{bull_avg:.4f},{bear_avg:.4f},0.0000,0.0000,{gross_profit_dollars:.2f},{gross_loss_dollars:.2f}")
    
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
    parser.add_argument("--retrain-interval", type=int, default=21,
                        help="HMM retrain every N trading days (21≈monthly). Use 1 for daily (live-trading accuracy).")
    parser.add_argument("--max-leverage", type=float, default=1.5,
                        help="Max total portfolio leverage (deployed $ / equity). Default 1.5× (150%% of capital).")
    parser.add_argument("--max-hold-days", type=int, default=10,
                        help="Max trading days to hold a momentum position before EOD force-close (0 = disabled). Default 10.")
    parser.add_argument("--cooldown-days", type=int, default=3,
                        help="Trading days a ticker is blacklisted after a stop-out. Default 3 (~1 week).")
    parser.add_argument("--loss-cooldown-threshold", type=float, default=-1.5,
                        help="PnL%% below which the extended cooldown fires instead of normal. Default -1.5%%.")
    parser.add_argument("--extended-cooldown-days", type=int, default=10,
                        help="Extended blacklist days applied after a large loss stop-out. Default 10.")
    parser.add_argument("--regime-dwell-days", type=int, default=3,
                        help="Min consecutive days a new HMM regime must hold before label switches (reduces lag). Default 3.")
    parser.add_argument("--slippage", type=float, default=0.0003,
                        help="Slippage per leg as a decimal fraction (0.0003 = 0.03%% = 3 bps). Default 0.0003.")
    parser.add_argument("--commission", type=float, default=0.0,
                        help="Commission per leg as a decimal fraction (0 = free like Alpaca). Default 0.0.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sparm = load_strategy_params(args.param_file)

    custom_params = {
        'stop_r': sparm['stop_r'],
        'target_r': sparm['target_r'],
        'risk_per_trade': sparm.get('risk_per_trade', 0.05),
        'target_trade_vol': 2.0,
        'max_total_leverage': args.max_leverage,
        'max_positions': 5,
        'max_hold_trading_days': args.max_hold_days,
        'ticker_cooldown_days': args.cooldown_days,
        'loss_cooldown_threshold': args.loss_cooldown_threshold,
        'ticker_cooldown_extended_days': args.extended_cooldown_days,
        'slippage_pct': args.slippage,
        'commission_pct': args.commission,
    }

    results = run_backtest_v2(
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        fetch_5m=not args.no_fetch,
        verbose=not args.quiet,
        always_long=args.long_only,
        params=custom_params,
        sparm=sparm,
        retrain_interval_days=args.retrain_interval,
        regime_dwell_days=args.regime_dwell_days,
    )
