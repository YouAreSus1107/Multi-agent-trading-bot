"""
War-Room Bot -- Quant Engine
Institutional-grade intraday feature engineering.
Augments (does NOT replace) the classic indicator stack.

Functions operate on pandas DataFrames from market_service.get_intraday_bars().
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("quant_engine")


# ──────────────────────────────────────────────────────────────────────────────
def compute_atr_14(df_15m: pd.DataFrame, period: int = 14) -> float:
    """
    Compute 14-period ATR on 5-minute bars using Wilder's smoothing.

    ATR measures true volatility in dollar terms (not %).
    Used to set stop_price = entry - 1.5 × ATR.

    Args:
        df_5m: DataFrame with [open, high, low, close, volume], datetime-indexed.
        period: Lookback (default 14 bars ≈ 70 minutes of 5-min data).

    Returns:
        ATR in dollars (float). Returns 0.0 if insufficient data.
    """
    if df_15m is None or len(df_15m) < period + 1:
        logger.debug(f"ATR: insufficient bars ({len(df_15m) if df_15m is not None else 0})")
        return 0.0

    h = df_15m["high"].values.astype(float)
    l = df_15m["low"].values.astype(float)
    c = df_15m["close"].values.astype(float)

    # True Range: max(H-L, |H-Cprev|, |L-Cprev|)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )

    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    # Wilder's smoothing: seed with simple mean, then exponential
    atr = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return round(float(atr), 4)


# ──────────────────────────────────────────────────────────────────────────────
# VWAP Z-Score — Statistical mean-reversion signal on 15-minute bars
# ──────────────────────────────────────────────────────────────────────────────

def compute_vwap_zscore(df_15m: pd.DataFrame, z_window: int = 20) -> dict:
    """
    Compute intraday VWAP + rolling Z-score of price deviation.

    VWAP = Σ(TP × Volume) / Σ(Volume)  where TP = (H+L+C)/3

    Z-score = (Price - VWAP) / rolling_std(Price - VWAP, window=z_window)

    Signals:
        Z > +2.5  → statistically overbought (mean-reversion SHORT setup)
        Z < -2.5  → statistically oversold   (mean-reversion LONG setup)
        |Z| < 1.0 → price near VWAP          (trend continuation safe zone)

    Returns:
        {
            vwap: float,
            vwap_zscore: float,
            sigma: float,       # rolling std of price deviation
            signal: str,        # "overbought" | "oversold" | "neutral"
            price_vs_vwap: str, # "above" | "below"
        }
    """
    empty = {"vwap": 0.0, "vwap_zscore": 0.0, "sigma": 0.0,
             "signal": "neutral", "price_vs_vwap": "unknown"}

    if df_15m is None or len(df_15m) < z_window + 5:
        return empty

    try:
        df = df_15m.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        # Cumulative VWAP (intraday — resets each day handled by slicing the day)
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_tp_vol"] = (df["typical_price"] * df["volume"]).cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
        df["vwap"] = df["vwap"].ffill()

        # Deviation of close from VWAP
        df["deviation"] = df["close"] - df["vwap"]

        # Rolling standard deviation of deviation
        # VWAP bands typically use cumulative standard deviation from the start of the day
        df["variance"] = (df["volume"] * (df["typical_price"] - df["vwap"])**2).cumsum() / df["cum_vol"].replace(0, np.nan)
        df["sigma"] = np.sqrt(df["variance"])
        df["zscore"] = df["deviation"] / df["sigma"].replace(0, np.nan)

        last = df.iloc[-1]
        vwap_val = float(last["vwap"]) if not np.isnan(last["vwap"]) else 0.0
        z = float(last["zscore"]) if not np.isnan(last["zscore"]) else 0.0
        sigma = float(last["sigma"]) if not np.isnan(last["sigma"]) else 0.0
        close = float(last["close"])

        if z > 2.5:
            signal = "overbought"
        elif z < -2.5:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "vwap": round(vwap_val, 2),
            "vwap_zscore": round(z, 3),
            "sigma": round(sigma, 4),
            "signal": signal,
            "price_vs_vwap": "above" if close > vwap_val else "below",
        }

    except Exception as e:
        logger.warning(f"VWAP Z-score computation failed: {e}")
        return empty


# ──────────────────────────────────────────────────────────────────────────────
# Volume Delta / Microstructure — 15-minute bars
# ──────────────────────────────────────────────────────────────────────────────

def compute_volume_delta(df_15m: pd.DataFrame, sma_window: int = 20) -> dict:
    """
    Decompose each 1-min candle into buy vs sell volume using price position.

    buy_vol  per bar = Volume × (Close - Low)  / (High - Low)
    sell_vol per bar = Volume × (High - Close) / (High - Low)

    Smart bounce (strict institutional definition):
        (Close - Low) / (High - Low) > 0.75   AND
        Volume > SMA_20(Volume) × 1.5

    Returns:
        {
            buy_pressure: float,    # total buy vol over last 20 bars
            sell_pressure: float,   # total sell vol over last 20 bars
            delta_ratio: float,     # buy / (buy + sell), 0–1. > 0.55 = bullish
            smart_bounce: bool,
            last_bar_buy_pct: float # (C-L)/(H-L) of last bar, 0–1
        }
    """
    empty = {"buy_pressure": 0, "sell_pressure": 0, "delta_ratio": 0.5,
             "smart_bounce": False, "last_bar_buy_pct": 0.5}

    if df_15m is None or len(df_15m) < sma_window + 1:
        return empty

    try:
        df = df_15m.copy()
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)

        df["buy_frac"] = (df["close"] - df["low"]) / hl_range
        df["sell_frac"] = (df["high"] - df["close"]) / hl_range

        # Fill NaN (doji candles where H==L) with 0.5
        df["buy_frac"] = df["buy_frac"].fillna(0.5)
        df["sell_frac"] = df["sell_frac"].fillna(0.5)

        df["buy_vol"] = df["buy_frac"] * df["volume"]
        df["sell_vol"] = df["sell_frac"] * df["volume"]

        # Use last sma_window bars for aggregate pressure
        window = df.iloc[-sma_window:]
        buy_pressure = float(window["buy_vol"].sum())
        sell_pressure = float(window["sell_vol"].sum())
        total = buy_pressure + sell_pressure
        delta_ratio = buy_pressure / total if total > 0 else 0.5

        # Smart bounce check on last bar
        last = df.iloc[-1]
        last_buy_pct = float(last["buy_frac"])

        vol_sma = float(df["volume"].rolling(sma_window).mean().iloc[-1]) if len(df) >= sma_window else float(df["volume"].mean())
        last_vol = float(last["volume"])

        smart_bounce = (last_buy_pct > 0.75) and (last_vol > vol_sma * 1.5)

        return {
            "buy_pressure": round(buy_pressure, 0),
            "sell_pressure": round(sell_pressure, 0),
            "delta_ratio": round(delta_ratio, 3),
            "smart_bounce": smart_bounce,
            "last_bar_buy_pct": round(last_buy_pct, 3),
        }

    except Exception as e:
        logger.warning(f"Volume delta computation failed: {e}")
        return empty


# ──────────────────────────────────────────────────────────────────────────────
# Master quant profile builder
# ──────────────────────────────────────────────────────────────────────────────

def build_quant_metrics(df_15m: pd.DataFrame) -> dict:
    """
    Compute all three quant metrics and derive signal labels for hybrid scoring.

    Returns a single dict to be merged into the technical profile.
    The hybrid TechnicalAnalyst will call this after computing classic indicators.

    Signal labels added (for research_team prompts):
        VWAP_OVERBOUGHT   → Z > 2.5  (mean-reversion short zone, reject buys)
        VWAP_OVERSOLD     → Z < -2.5 (mean-reversion long opportunity)
        VWAP_NEUTRAL      → |Z| < 1.0 (trend continuation safe zone)
        SMART_BOUNCE      → strict institutional smart bounce detected
        VOLUME_BULLISH    → delta_ratio > 0.55
        VOLUME_BEARISH    → delta_ratio < 0.45
    """
    if df_15m is None or df_15m.empty:
        return {
            "atr_5m": 0.0, "initial_stop": 0.0, "target_2r": 0.0,
            "vwap": 0.0, "vwap_zscore": 0.0, "vwap_sigma": 0.0, "vwap_signal": "neutral", "price_vs_vwap": "unknown",
            "delta_ratio": 0.5, "buy_pressure": 0, "sell_pressure": 0, "smart_bounce": False, "last_bar_buy_pct": 0.5,
            "volume_ratio": 1.0,
            "mtf_total_pos": 0, "tsl_1": 0.0, "rsi_14": 50.0,
            "quant_signals": []
        }

    last_row = df_15m.iloc[-1]
    
    # Extract MTF early if precomputed (backtesting fast-path)
    mtf_total_pos = int(last_row.get("mtf_total_pos", 0)) if "mtf_total_pos" in df_15m.columns else 0
    tsl_1 = float(last_row.get("tsl_1", 0.0)) if "tsl_1" in df_15m.columns else 0.0
    
    if "atr_5m" in df_15m.columns and "vwap_zscore" in df_15m.columns:
        # Fast-Path: Backtester precomputed caching to bypass loop execution scaling
        atr_val = float(last_row["atr_5m"]) if not pd.isna(last_row["atr_5m"]) else 0.0
        rsi_14_val = float(last_row["rsi_14"]) if "rsi_14" in df_15m.columns and not pd.isna(last_row["rsi_14"]) else 50.0
        
        vwap_data = {
            "vwap": float(last_row["vwap"]),
            "vwap_zscore": float(last_row["vwap_zscore"]),
            "sigma": float(last_row["sigma"]) if "sigma" in df_15m.columns else 0.0,
            "signal": "overbought" if last_row["vwap_zscore"] > 2.5 else "oversold" if last_row["vwap_zscore"] < -2.5 else "neutral",
            "price_vs_vwap": "above" if last_row["close"] > last_row["vwap"] else "below"
        }
        
        delta_data = {
            "buy_pressure": float(last_row["buy_pressure"]),
            "sell_pressure": float(last_row["sell_pressure"]),
            "delta_ratio": float(last_row["delta_ratio"]),
            "smart_bounce": bool(last_row["smart_bounce"]),
            "last_bar_buy_pct": float(last_row["last_bar_buy_pct"])
        }
        volume_ratio_val = float(last_row.get("volume_ratio", 1.0)) if "volume_ratio" in df_15m.columns else 1.0
    else:
        # Live execution or explicit manual pass
        atr_val = compute_atr_14(df_15m)
        vwap_data = compute_vwap_zscore(df_15m)
        delta_data = compute_volume_delta(df_15m)
        # Compute RSI-14 live (Wilder's EWM)
        rsi_14_val = 50.0
        if len(df_15m) >= 15:
            delta = df_15m["close"].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else float('inf')
            rsi_14_val = float(100 - (100 / (1 + rs))) if not np.isnan(rs) else 50.0
        
        # Compute volume_ratio for live execution
        if len(df_15m) >= 20:
            vol_sma_live = float(df_15m["volume"].rolling(20).mean().iloc[-1])
            volume_ratio_val = round(float(df_15m["volume"].iloc[-1]) / max(vol_sma_live, 1.0), 3)
        else:
            volume_ratio_val = 1.0

        # Inject MTF calculation into the live flow
        if "mtf_total_pos" not in df_15m.columns:
            df_mtf = precompute_mtf_tsl_for_ticker(df_15m)
            last_mtf = df_mtf.iloc[-1]
            mtf_total_pos = int(last_mtf.get("mtf_total_pos", 0))
            tsl_1 = float(last_mtf.get("tsl_1", 0.0))
        else:
            mtf_total_pos = int(last_row.get("mtf_total_pos", 0))
            tsl_1 = float(last_row.get("tsl_1", 0.0))

    # Derive current price
    current_price = 0.0
    if df_15m is not None and not df_15m.empty:
        current_price = float(df_15m["close"].iloc[-1])

    # Dynamic stop and target from ATR
    initial_stop = round(current_price - 1.5 * atr_val, 2) if atr_val > 0 and current_price > 0 else 0.0
    target_2r    = round(current_price + 3.0 * atr_val, 2) if atr_val > 0 and current_price > 0 else 0.0

    # Signal labels
    quant_signals = []
    z = vwap_data.get("vwap_zscore", 0)
    if z > 2.5:
        quant_signals.append("VWAP_OVERBOUGHT")
    elif z < -2.5:
        quant_signals.append("VWAP_OVERSOLD")
    elif abs(z) < 1.0:
        quant_signals.append("VWAP_NEUTRAL")

    if delta_data.get("smart_bounce"):
        quant_signals.append("SMART_BOUNCE")

    delta_ratio = delta_data.get("delta_ratio", 0.5)
    if delta_ratio > 0.55:
        quant_signals.append("VOLUME_BULLISH")
    elif delta_ratio < 0.45:
        quant_signals.append("VOLUME_BEARISH")

    return {
        # ATR
        "atr_5m": atr_val,
        "initial_stop": initial_stop,
        "target_2r": target_2r,
        # VWAP Z-Score
        "vwap": vwap_data.get("vwap", 0.0),
        "vwap_zscore": vwap_data.get("vwap_zscore", 0.0),
        "vwap_sigma": vwap_data.get("sigma", 0.0),
        "vwap_signal": vwap_data.get("signal", "neutral"),
        "price_vs_vwap": vwap_data.get("price_vs_vwap", "unknown"),
        # Volume Delta
        "delta_ratio": delta_ratio,
        "buy_pressure": delta_data.get("buy_pressure", 0),
        "sell_pressure": delta_data.get("sell_pressure", 0),
        "smart_bounce": delta_data.get("smart_bounce", False),
        "last_bar_buy_pct": delta_data.get("last_bar_buy_pct", 0.5),
        # Volume Ratio (current bar volume / 20-bar SMA volume)
        "volume_ratio": volume_ratio_val,
        # RSI
        "rsi_14": rsi_14_val,
        # MTF Setup
        "mtf_total_pos": mtf_total_pos,
        "tsl_1": tsl_1,
        # Signal labels for research prompts
        "quant_signals": quant_signals,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pine Script: MTF Trailing SL [QuantNomad]
# ──────────────────────────────────────────────────────────────────────────────

def _calc_tsl_series(df: pd.DataFrame, atr_length: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """Helper to compute the TSL on a given resolution timeframe."""
    if len(df) < atr_length + 1:
        return pd.Series(np.nan, index=df.index)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    tr = np.zeros(len(df))
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )

    atr = np.zeros(len(df))
    atr[atr_length] = np.mean(tr[1:atr_length+1])
    for i in range(atr_length + 1, len(df)):
        atr[i] = (atr[i-1] * (atr_length - 1) + tr[i]) / atr_length

    sl_val = atr * atr_mult

    pos = np.zeros(len(df))
    tsl = np.zeros(len(df))

    # Calculate sequentially as per Pine Script logic
    for i in range(1, len(df)):
        prev_pos = pos[i-1]
        prev_tsl = tsl[i-1]

        long_signal = (prev_pos != 1) and (highs[i] > prev_tsl)
        short_signal = (prev_pos != -1) and (lows[i] < prev_tsl)

        if short_signal:
            curr_tsl = highs[i] + sl_val[i]
        elif long_signal:
            curr_tsl = lows[i] - sl_val[i]
        elif prev_pos == 1:
            curr_tsl = max(lows[i] - sl_val[i], prev_tsl)
        elif prev_pos == -1:
            curr_tsl = min(highs[i] + sl_val[i], prev_tsl)
        else:
            curr_tsl = prev_tsl

        if long_signal:
            pos[i] = 1
        elif short_signal:
            pos[i] = -1
        else:
            pos[i] = prev_pos

        tsl[i] = curr_tsl

    return pd.Series(tsl, index=df.index)


def precompute_mtf_tsl_for_ticker(df_5m: pd.DataFrame, atr_length: int = 14, atr_mult: float = 2.0) -> pd.DataFrame:
    """
    Computes QuantNomad's MTF Trailing SL upfront using localized pandas resampling.
    TF1 = 5m (base), TF2 = 120m, TF3 = 180m, TF4 = 240m.
    Adds `mtf_total_pos` and `tsl_1` columns directly to the DataFrame.
    """
    df = df_5m.copy()
    if df.empty or len(df) < atr_length + 2:
        df["mtf_total_pos"] = 0
        df["tsl_1"] = 0.0
        return df

    try:
        # Build Higher Timeframes using resample
        df_120m = df.resample('120min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        df_180m = df.resample('180min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        df_240m = df.resample('240min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        # Compute TSL for each
        tsl_1 = _calc_tsl_series(df, atr_length, atr_mult)
        tsl_2 = _calc_tsl_series(df_120m, atr_length, atr_mult)
        tsl_3 = _calc_tsl_series(df_180m, atr_length, atr_mult)
        tsl_4 = _calc_tsl_series(df_240m, atr_length, atr_mult)

        # Re-align back to 5m timeframe using forward fill mapping
        df["tsl_1"] = tsl_1
        
        tsl_2_df = pd.DataFrame({"tsl_2": tsl_2})
        tsl_3_df = pd.DataFrame({"tsl_3": tsl_3})
        tsl_4_df = pd.DataFrame({"tsl_4": tsl_4})
        
        df = pd.merge_asof(df, tsl_2_df, left_index=True, right_index=True, direction='backward')
        df = pd.merge_asof(df, tsl_3_df, left_index=True, right_index=True, direction='backward')
        df = pd.merge_asof(df, tsl_4_df, left_index=True, right_index=True, direction='backward')
        
        # Fill missing early NaNs
        df = df.ffill().bfill()

        # Determine positions logic from PineScript:
        def calc_pos(series_low, series_high, series_tsl):
            pos = np.zeros(len(df))
            lows = series_low.values
            highs = series_high.values
            tsls = series_tsl.values
            for i in range(1, len(df)):
                if lows[i] <= tsls[i]:
                    pos[i] = -1
                elif highs[i] >= tsls[i]:
                    pos[i] = 1
                else:
                    pos[i] = pos[i-1]
            return pos

        pos1 = calc_pos(df["low"], df["high"], df["tsl_1"])
        pos2 = calc_pos(df["low"], df["high"], df["tsl_2"])
        pos3 = calc_pos(df["low"], df["high"], df["tsl_3"])
        pos4 = calc_pos(df["low"], df["high"], df["tsl_4"])

        df["mtf_total_pos"] = pos1 + pos2 + pos3 + pos4
        
        return df
    except Exception as e:
        logger.warning(f"MTF TSL computation failed: {e}")
        df["mtf_total_pos"] = 0
        df["tsl_1"] = 0.0
        return df

def precompute_quant_metrics_for_ticker(df_15m: pd.DataFrame, atr_period: int = 14, vwap_z_window: int = 20, volume_sma_window: int = 20) -> pd.DataFrame:
    """
    Vectorized O(N) hardware-accelerated computation of ALL classic quant metrics over 
    the entire dataset upfront. Avoids slicing 20-day Pandas loops.
    """
    df = df_15m.copy()
    if df.empty:
        return df

    try:
        # --- 1. ATR Block (Vectorized Wilder's) ---
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr_series = pd.Series(np.insert(tr, 0, np.nan), index=df.index)
        # EWM gives an exact vectorized match to Wilder's iterative loop
        df["atr_5m"] = tr_series.ewm(alpha=1/atr_period, adjust=False).mean().round(4)
        df["atr_5m"] = df["atr_5m"].fillna(0.0)

        # --- 2. VWAP Z-score Block ---
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
        # Compute intraday cumsum resetting per day boundary automatically via grouping
        df['date'] = df.index.date
        
        df["cum_vol"] = df.groupby('date')["volume"].cumsum()
        tp_vol = df["typical_price"] * df["volume"]
        df["cum_tp_vol"] = tp_vol.groupby(df['date']).cumsum()
        
        df["vwap"] = (df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)).ffill().fillna(df["close"])
        df["deviation"] = df["close"] - df["vwap"]
        # Volume-weighted cumulative variance (matches live compute_vwap_zscore formula)
        # variance = Σ(V × (TP - VWAP)²) / Σ(V), reset per day via groupby
        vw_sq = df["volume"] * (df["typical_price"] - df["vwap"]) ** 2
        df["sigma"] = np.sqrt(
            vw_sq.groupby(df['date']).cumsum() / df["cum_vol"].replace(0, np.nan)
        )
        df["vwap_zscore"] = (df["deviation"] / df["sigma"].replace(0, np.nan)).round(3).fillna(0.0)
        
        # --- 3. Volume Delta Microstructure Block ---
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["buy_frac"] = ((df["close"] - df["low"]) / hl_range).fillna(0.5)
        df["sell_frac"] = ((df["high"] - df["close"]) / hl_range).fillna(0.5)
        df["buy_vol"] = df["buy_frac"] * df["volume"]
        df["sell_vol"] = df["sell_frac"] * df["volume"]
        
        df["buy_pressure"] = df["buy_vol"].rolling(window=volume_sma_window, min_periods=1).sum().round(0)
        df["sell_pressure"] = df["sell_vol"].rolling(window=volume_sma_window, min_periods=1).sum().round(0)
        
        total_pressure = df["buy_pressure"] + df["sell_pressure"]
        df["delta_ratio"] = (df["buy_pressure"] / total_pressure.replace(0, 1.0)).round(3)
        
        vol_sma = df["volume"].rolling(volume_sma_window, min_periods=1).mean()
        df["smart_bounce"] = (df["buy_frac"] > 0.75) & (df["volume"] > vol_sma * 1.5)
        df["last_bar_buy_pct"] = df["buy_frac"].round(3)
        df["volume_ratio"] = (df["volume"] / vol_sma.replace(0, 1.0)).round(3)

        # --- 4. RSI-14 Block (Wilder's EWM — vectorized) ---
        close_delta = df["close"].diff()
        gain = close_delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-close_delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - (100 / (1 + rs))).round(2).fillna(50.0)

        df.drop(columns=['date', 'typical_price', 'cum_vol', 'cum_tp_vol', 'deviation', 'buy_frac', 'sell_frac', 'buy_vol', 'sell_vol'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        logger.warning(f"Vectorized precomputation failed: {e}")
        return df
