"""
V2 Data Loader — 3-Layer Backtesting Architecture
===================================================
Layer 1: Parse wide-format S&P 500 daily CSV, compute Universe Ranking
         (Alpha, Beta, Volatility, Momentum), and select Top N tickers daily.
         Fetch & cache Alpaca 5-minute intraday data for selected tickers.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache_5m")
SP500_CSV = os.path.join(DATA_DIR, "SnP_daily_update.csv")

# Inverse ETFs to include in the universe for bear-regime trading
INVERSE_ETFS = ["SH", "SDS", "SPXU", "PSQ", "QID", "SQQQ", "DOG", "DXD", "SDOW",
                "RWM", "TZA", "SRTY", "SPXS", "TECS", "SOXS"]

# S&P 500 proxy for regime detection + benchmark
SPY_TICKER = "SPY"


# ──────────────────────────────────────────────────────────────────────────────
# 1. PARSE THE WIDE-FORMAT KAGGLE CSV
# ──────────────────────────────────────────────────────────────────────────────

_PARSED_DAILY_CACHE = None

def parse_sp500_daily(csv_path: str = SP500_CSV, fetch_yfinance: bool = True) -> dict[str, pd.DataFrame]:
    """
    Parse the Kaggle S&P 500 wide-format CSV into per-ticker DataFrames.
    
    The CSV layout:
      Row 0: "Ticker", <ticker1>, <ticker2>, ...  (repeated over Close/High/Low/Open/Volume groups)
      Row 1: "Date", NaN, NaN, ...
      Row 2+: dates, values, values, ...
      
    Columns are grouped: Close (503 cols), High (503), Low (503), Open (503), Volume (503)
    
    Returns:
        dict[ticker_str, pd.DataFrame] with columns [open, high, low, close, volume]
        indexed by DatetimeIndex.
    """
    global _PARSED_DAILY_CACHE
    if _PARSED_DAILY_CACHE is not None:
        return _PARSED_DAILY_CACHE

    print("Parsing S&P 500 daily CSV (this may take a moment)...")
    df_raw = pd.read_csv(csv_path, header=0)
    
    # Row 0 contains ticker names — read it
    ticker_row = df_raw.iloc[0].values  # e.g. ["Ticker", "A", "AAPL", ...]
    
    # Identify column groups by their header prefix
    col_headers = df_raw.columns.tolist()  # ["Price", "Close", "Close.1", ..., "High", "High.1", ...]
    
    # Build mapping: column_name -> (field_type, ticker)
    field_groups = {}  # field_type -> list of (col_index, ticker)
    current_field = None
    
    for i, col_name in enumerate(col_headers):
        if i == 0:
            continue  # skip "Price" column (contains dates)
        
        # Determine field type from column header
        base = col_name.split('.')[0]  # "Close", "High", "Low", "Open", "Volume"
        ticker = str(ticker_row[i])
        
        if ticker == 'nan' or ticker == 'Ticker':
            continue
            
        if base not in field_groups:
            field_groups[base] = []
        field_groups[base].append((i, ticker))
    
    # Data starts at row 2 (skip ticker row and date header row)
    data_df = df_raw.iloc[2:].copy()
    data_df.reset_index(drop=True, inplace=True)
    
    # Column 0 = dates
    dates = pd.to_datetime(data_df.iloc[:, 0], errors='coerce')
    
    # Build per-ticker DataFrames
    result = {}
    
    # Get all unique tickers from the Close group
    if 'Close' not in field_groups:
        raise ValueError("No 'Close' columns found in CSV")
    
    close_tickers = {t: i for i, t in field_groups['Close']}
    
    for ticker in close_tickers:
        try:
            ticker_data = {'date': dates}
            
            for field_type in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if field_type not in field_groups:
                    continue
                # Find the column index for this ticker in this field group
                col_idx = None
                for idx, t in field_groups[field_type]:
                    if t == ticker:
                        col_idx = idx
                        break
                if col_idx is not None:
                    ticker_data[field_type.lower()] = pd.to_numeric(
                        data_df.iloc[:, col_idx], errors='coerce'
                    )
            if 'close' not in ticker_data:
                continue
            
            # Skip dual-class shares that cause severe API routing bugs
            if ticker in ["BF.B", "BF-B", "BRK.B", "BRK-B"]:
                continue
                
            tdf = pd.DataFrame(ticker_data)
            tdf.set_index('date', inplace=True)
            tdf.dropna(subset=['close'], inplace=True)
            tdf.sort_index(inplace=True)
            
            if len(tdf) > 0:
                result[ticker] = tdf
        except Exception:
            continue
    
    print(f"  Parsed {len(result)} tickers from S&P 500 daily CSV")
    
    # Auto-supplement ETF data (SPY + inverse ETFs not in the S&P 500 component list)
    result = supplement_etf_data(result, fetch_yfinance=fetch_yfinance)
    
    _PARSED_DAILY_CACHE = result
    
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 1b. SUPPLEMENT ETF DATA (SPY, Inverse ETFs) via yfinance
# ──────────────────────────────────────────────────────────────────────────────

def supplement_etf_data(
    all_tickers: dict[str, pd.DataFrame],
    extra_symbols: list[str] = None,
    fetch_yfinance: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for ETFs not present in the S&P 500 constituent CSV.
    Uses yfinance. Caches results locally.
    """
    if extra_symbols is None:
        extra_symbols = [SPY_TICKER] + INVERSE_ETFS
    
    # Only fetch symbols not already present
    missing = [s for s in extra_symbols if s not in all_tickers]
    
    if not missing:
        return all_tickers
    
    cache_file = os.path.join(DATA_DIR, "etf_daily_cache.parquet")
    
    # Try loading from cache first
    cached_data = {}
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_parquet(cache_file)
            if 'symbol' in cached_df.columns:
                for sym, grp in cached_df.groupby('symbol'):
                    grp = grp.drop(columns=['symbol']).copy()
                    grp.sort_index(inplace=True)
                    cached_data[sym] = grp
                
                # Check if all missing are in cache
                still_missing = [s for s in missing if s not in cached_data]
                if not still_missing:
                    for sym, df in cached_data.items():
                        if sym not in all_tickers:
                            all_tickers[sym] = df
                    print(f"  Loaded {len(cached_data)} ETFs from cache (SPY + Inverse ETFs)")
                    return all_tickers
        except Exception:
            pass
    
    if not fetch_yfinance:
        # Load whatever is cached, skip yfinance entirely
        for sym, df in cached_data.items():
            if sym not in all_tickers:
                all_tickers[sym] = df
        if cached_data:
            print(f"  Loaded {len(cached_data)} ETFs from cache (--no-fetch mode)")
        else:
            print("  WARNING: No ETF cache found and --no-fetch is set. Regime may fail.")
        return all_tickers
    
    # Download from yfinance
    print(f"  Downloading ETF daily data for {len(missing)} symbols via yfinance...")
    try:
        import yfinance as yf
        
        # Determine date range from existing data
        sample_df = list(all_tickers.values())[0]
        start_date = sample_df.index.min().strftime("%Y-%m-%d")
        end_date = sample_df.index.max().strftime("%Y-%m-%d")
        
        all_etf_dfs = []
        
        for sym in missing:
            try:
                ticker_obj = yf.Ticker(sym)
                need_adjust = sym not in INVERSE_ETFS
                hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=need_adjust)
                
                if hist.empty:
                    continue
                
                df = pd.DataFrame({
                    'open': hist['Open'],
                    'high': hist['High'],
                    'low': hist['Low'],
                    'close': hist['Close'],
                    'volume': hist['Volume']
                })
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.sort_index(inplace=True)
                df.dropna(subset=['close'], inplace=True)
                
                if len(df) > 0:
                    all_tickers[sym] = df
                    cache_df = df.copy()
                    cache_df['symbol'] = sym
                    all_etf_dfs.append(cache_df)
                    
            except Exception as e:
                print(f"    Failed to download {sym}: {e}")
        
        # Save cache
        if all_etf_dfs:
            try:
                combined = pd.concat(all_etf_dfs)
                combined.to_parquet(cache_file)
                print(f"  Cached {len(all_etf_dfs)} ETFs to disk")
            except Exception:
                pass
        
        added = [s for s in missing if s in all_tickers]
        print(f"  Added {len(added)} ETFs: {', '.join(added[:8])}{'...' if len(added) > 8 else ''}")
        
    except ImportError:
        print("  WARNING: yfinance not installed. Cannot supplement ETF data.")
    except Exception as e:
        print(f"  ETF supplement error: {e}")
    
    return all_tickers


# ──────────────────────────────────────────────────────────────────────────────
# 2. QUANT RANKING ENGINE (Alpha, Beta, Volatility, Momentum)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_alpha_beta(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 30
) -> tuple[float, float]:
    """
    Compute rolling Alpha and Beta vs benchmark (SPY) using the last `window` days.
    Alpha = ticker_return - beta * benchmark_return  (annualised excess return)
    Beta  = cov(ticker, benchmark) / var(benchmark)
    """
    if len(ticker_returns) < window or len(benchmark_returns) < window:
        return 0.0, 1.0
    
    t_ret = ticker_returns.iloc[-window:]
    b_ret = benchmark_returns.iloc[-window:]
    
    # Align
    aligned = pd.DataFrame({'t': t_ret, 'b': b_ret}).dropna()
    if len(aligned) < 10:
        return 0.0, 1.0
    
    cov_matrix = np.cov(aligned['t'].values, aligned['b'].values, ddof=0)
    var_b = cov_matrix[1, 1]
    
    if var_b == 0:
        return 0.0, 1.0
    
    beta = cov_matrix[0, 1] / var_b
    alpha = (aligned['t'].mean() - beta * aligned['b'].mean()) * 252  # Annualised
    
    return float(alpha), float(beta)


def compute_momentum_score(closes: pd.Series, periods: list[int] = None) -> float:
    """
    Multi-timeframe momentum: weighted average of rate-of-change across periods.
    Higher = stronger upward momentum.
    """
    if periods is None:
        periods = [5, 10, 20, 60]
    
    weights = [0.4, 0.3, 0.2, 0.1]  # More weight on recent momentum
    score = 0.0
    total_weight = 0.0
    
    for period, weight in zip(periods, weights):
        if len(closes) > period and closes.iloc[-period] > 0:
            roc = (closes.iloc[-1] / closes.iloc[-period] - 1) * 100
            score += roc * weight
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0


def rank_universe(
    all_tickers: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    lookback_days: int = 252,  # 1 year for warmup context
    ranking_window: int = 30,  # Rolling window for alpha/beta
    top_n: int = 10
) -> list[dict]:
    """
    Rank the entire S&P 500 universe for a given trading day.
    
    Scoring formula (weighted composite):
      - Alpha (40%): Excess return vs SPY
      - Momentum (30%): Multi-timeframe rate of change  
      - Idiosyncratic Vol (20%): High vol = more opportunity
      - Low Beta preference (10%): Lower correlation = more alpha
      
    Returns:
        List of top_n dicts: [{ticker, alpha, beta, volatility, momentum, composite_score}, ...]
    """
    # Get SPY benchmark data
    if SPY_TICKER not in all_tickers:
        print("  WARNING: SPY not in dataset, using first ticker as proxy")
        spy_data = list(all_tickers.values())[0]
    else:
        spy_data = all_tickers[SPY_TICKER]
    
    # Slice to lookback window ending at as_of_date
    start_date = as_of_date - pd.Timedelta(days=lookback_days)
    spy_slice = spy_data.loc[start_date:as_of_date]
    
    if len(spy_slice) < 30:
        return []
    
    spy_returns = spy_slice['close'].pct_change().dropna()
    
    candidates = []
    
    # Tickers to exclude from ranking (benchmarks + inverse ETFs — traded separately)
    excluded = {SPY_TICKER, 'SPX', 'QQQ', 'IWM', 'GLD', 'DIA'}
    excluded.update(INVERSE_ETFS)
    
    for ticker, tdf in all_tickers.items():
        if ticker in excluded:
            continue
            
        t_slice = tdf.loc[start_date:as_of_date]
        
        if len(t_slice) < 60:  # Need minimum data
            continue
        
        # Skip low-price / penny stocks
        last_price = t_slice['close'].iloc[-1]
        if last_price < 5.0 or np.isnan(last_price):
            continue
        
        # Skip low volume (if volume data available)
        if 'volume' in t_slice.columns:
            avg_vol = t_slice['volume'].iloc[-20:].mean()
            if not np.isnan(avg_vol) and avg_vol < 100000:
                continue
        
        t_returns = t_slice['close'].pct_change().dropna()
        
        # 1. Alpha & Beta
        alpha, beta = compute_rolling_alpha_beta(t_returns, spy_returns, ranking_window)
        
        # 2. Volatility (annualised)
        volatility = float(t_returns.iloc[-ranking_window:].std() * np.sqrt(252)) if len(t_returns) >= ranking_window else 0.0
        
        # 3. Momentum
        momentum = compute_momentum_score(t_slice['close'])
        
        # 4. Composite Score (weighted)
        # Normalise components later via ranking, for now use raw
        composite = (
            0.40 * alpha +          # Higher alpha = better
            0.30 * momentum +       # Stronger momentum = better
            0.20 * volatility +     # Higher idiosyncratic vol = more opportunity
            0.10 * (2.0 - abs(beta))  # Prefer beta close to 0 (idiosyncratic)
        )
        
        candidates.append({
            'ticker': ticker,
            'alpha': round(alpha, 4),
            'beta': round(beta, 4),
            'volatility': round(volatility, 4),
            'momentum': round(momentum, 4),
            'composite_score': round(composite, 4),
            'last_price': round(last_price, 2)
        })
    
    # Sort by composite score descending and return top N
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    return candidates[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# 3. ALPACA 5-MINUTE DATA FETCHER + CACHE
# ──────────────────────────────────────────────────────────────────────────────

def fetch_5m_bars_for_day(
    tickers: list[str],
    trade_date: datetime,
    cache_dir: str = CACHE_DIR,
    cache_only: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Fetch 5-minute bars from Alpaca for a list of tickers on a specific trading day.
    Uses local file caching to avoid redundant API calls.
    
    Args:
        cache_only: If True, only load from disk cache (skip Alpaca API download).
    
    Returns:
        dict[ticker, pd.DataFrame] with columns [open, high, low, close, volume]
        indexed by UTC timestamp.
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = trade_date.strftime("%Y-%m-%d")
    
    result = {}
    tickers_to_fetch = []
    
    # 1. Check cache first
    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}_{date_str}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    result[ticker] = df
                    continue
            except Exception:
                pass
        tickers_to_fetch.append(ticker)
    
    # 2. Fetch missing tickers from Alpaca (skip if cache_only mode)
    if tickers_to_fetch and not cache_only:
        try:
            from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            
            start_dt = trade_date.replace(hour=9, minute=30, second=0, tzinfo=timezone.utc)
            end_dt = trade_date.replace(hour=20, minute=0, second=0, tzinfo=timezone.utc)
            
            tf_5m = TimeFrame(amount=5, unit=TimeFrameUnit.Minute)
            
            # Fetch in batches to respect rate limits
            batch_size = 10
            for batch_start in range(0, len(tickers_to_fetch), batch_size):
                batch = tickers_to_fetch[batch_start:batch_start + batch_size]
                
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=tf_5m,
                        start=start_dt,
                        end=end_dt,
                        feed="iex"
                    )
                    bars = client.get_stock_bars(request).df
                    
                    if not bars.empty:
                        bars.reset_index(inplace=True)
                        for ticker in batch:
                            ticker_bars = bars[bars['symbol'] == ticker].copy()
                            if not ticker_bars.empty:
                                ticker_bars.set_index('timestamp', inplace=True)
                                ticker_bars.sort_index(inplace=True)
                                
                                # Cache to disk
                                cache_file = os.path.join(cache_dir, f"{ticker}_{date_str}.parquet")
                                ticker_bars.to_parquet(cache_file)
                                result[ticker] = ticker_bars
                                
                except Exception as e:
                    print(f"  Alpaca fetch error for batch {batch}: {e}")
                    
        except ImportError:
            print("  WARNING: Alpaca SDK not available. Cannot fetch 5m data.")
        except Exception as e:
            print(f"  Alpaca client error: {e}")
    
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. DAILY SIMULATION DRIVER
# ──────────────────────────────────────────────────────────────────────────────

def get_trading_days(
    all_tickers: dict[str, pd.DataFrame],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31"
) -> list[pd.Timestamp]:
    """
    Extract valid trading days from the SPY data within the specified range.
    """
    if SPY_TICKER in all_tickers:
        spy = all_tickers[SPY_TICKER]
    else:
        spy = list(all_tickers.values())[0]
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    trading_days = spy.loc[start:end].index.tolist()
    return trading_days


_DAILY_SELECTION_CACHE = {}

def run_daily_selection(
    all_tickers: dict[str, pd.DataFrame],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    warmup_days: int = 252,
    top_n: int = 10,
    fetch_5m: bool = True,
    verbose: bool = True
) -> list[dict]:
    """
    Main entry point for Layer 1: run the daily universe selection engine.
    
    For each trading day in [start_date, end_date]:
      1. Rank the S&P 500 universe using the previous `warmup_days` of data
      2. Select top_n tickers
      3. Optionally fetch 5m data for those tickers
      
    Returns:
        List of daily snapshots: [{
            date, regime (from Layer 2), 
            top_tickers: [{ticker, alpha, beta, ...}],
            intraday_data: {ticker: DataFrame} (if fetch_5m=True)
        }, ...]
    """
    from backtest.regime_v3 import HMMRegimeModel
    
    global _DAILY_SELECTION_CACHE
    cache_key = (start_date, end_date, warmup_days, top_n, fetch_5m)
    if cache_key in _DAILY_SELECTION_CACHE:
        if verbose:
            print(f"  [CACHE] Restoring {len(_DAILY_SELECTION_CACHE[cache_key])} daily rankings from memory...")
        return _DAILY_SELECTION_CACHE[cache_key]
        
    trading_days = get_trading_days(all_tickers, start_date, end_date)
    
    if not trading_days:
        print(f"No trading days found between {start_date} and {end_date}")
        return []
    
    # Compute market regime for the entire period using SPY
    if SPY_TICKER in all_tickers:
        spy_df = all_tickers[SPY_TICKER]
        from backtest.regime_v3 import HMMRegimeModel
        # Train on history up to start_date, inference strictly out-of-sample thereafter.
        regime_df = HMMRegimeModel().compute_regime(spy_df, train_end=start_date)
    else:
        regime_df = None
    
    daily_snapshots = []
    
    for i, trade_date in enumerate(trading_days):
        # 1. Rank universe using data UP TO (but not including) trade_date
        # This prevents look-ahead bias
        top_picks = rank_universe(
            all_tickers,
            as_of_date=trade_date - pd.Timedelta(days=1),
            lookback_days=warmup_days,
            ranking_window=30,
            top_n=top_n
        )
        
        if not top_picks:
            continue
        
        # 2. Determine regime for this day
        if regime_df is not None and trade_date in regime_df.index:
            regime = regime_df.loc[trade_date, 'regime']
        else:
            regime = 'bull'  # Default to bull if no regime data
        
        # 3. Fetch 5m data for selected tickers (+ SPY for reference)
        selected_tickers = [p['ticker'] for p in top_picks] + [SPY_TICKER]
        
        # In bear regime, add inverse ETFs to the selection pool
        if regime == 'bear':
            selected_tickers.extend(INVERSE_ETFS[:5])  # Top 5 inverse ETFs
        
        intraday_data = {}
        intraday_data = fetch_5m_bars_for_day(
                selected_tickers,
                trade_date.to_pydatetime() if hasattr(trade_date, 'to_pydatetime') else trade_date,
                cache_only=not fetch_5m  # When --no-fetch, load cache only (skip Alpaca)
            )
        
        snapshot = {
            'date': trade_date,
            'regime': regime,
            'top_tickers': top_picks,
            'intraday_data': intraday_data,
            'selected_tickers': selected_tickers
        }
        daily_snapshots.append(snapshot)
        
        if verbose and (i % 20 == 0 or i == len(trading_days) - 1):
            ticker_str = ", ".join([p['ticker'] for p in top_picks[:5]])
            print(f"  [{trade_date.strftime('%Y-%m-%d')}] Regime={regime.upper():4s} | Top 5: {ticker_str}")
    
    print(f"\n  Daily selection complete: {len(daily_snapshots)} trading days processed")
    _DAILY_SELECTION_CACHE[cache_key] = daily_snapshots
    return daily_snapshots


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V2 Data Loader: S&P 500 Universe Ranking")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top tickers to select daily")
    parser.add_argument("--no-fetch", action="store_true", help="Skip 5m data fetching (ranking only)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    print("=" * 70)
    print("V2 BACKTEST DATA LOADER — Layer 1: Universe Ranking")
    print("=" * 70)
    
    all_tickers = parse_sp500_daily()
    
    print(f"\n  Tickers loaded: {len(all_tickers)}")
    if SPY_TICKER in all_tickers:
        spy = all_tickers[SPY_TICKER]
        print(f"  SPY date range: {spy.index.min().strftime('%Y-%m-%d')} to {spy.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n  Running daily selection from {args.start} to {args.end}...")
    snapshots = run_daily_selection(
        all_tickers,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        fetch_5m=not args.no_fetch,
        verbose=args.verbose
    )
    
    print(f"\n  Total trading days: {len(snapshots)}")
    if snapshots:
        bull_days = sum(1 for s in snapshots if s['regime'] == 'bull')
        bear_days = sum(1 for s in snapshots if s['regime'] == 'bear')
        print(f"  Bull days: {bull_days} | Bear days: {bear_days}")
