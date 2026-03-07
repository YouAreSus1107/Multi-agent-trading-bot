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
        df["sigma"] = df["deviation"].rolling(window=z_window, min_periods=5).std()
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
    atr_val = compute_atr_14(df_15m)
    vwap_data = compute_vwap_zscore(df_15m)
    delta_data = compute_volume_delta(df_15m)

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
        # Signal labels for research prompts
        "quant_signals": quant_signals,
    }
