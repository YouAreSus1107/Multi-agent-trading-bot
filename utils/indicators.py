"""
War-Room Bot -- Technical Indicators (Extended)
RSI, Bollinger Bands, MACD, EMA Crossover, ATR, Volume Analysis,
Sharpe Ratio, Kelly Criterion, and Composite Trend Score.
"""

import numpy as np
from typing import Tuple


def compute_rsi(prices: list[float], period: int = 14) -> float:
    """Compute RSI (0-100). >70 overbought, <30 oversold."""
    if len(prices) < period + 1:
        return 50.0

    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


def compute_bollinger_bands(
    prices: list[float], period: int = 20, std_dev: float = 2.0
) -> Tuple[float, float, float]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    if len(prices) < period:
        mid = np.mean(prices) if prices else 0
        return (mid, mid, mid)

    prices_arr = np.array(prices[-period:], dtype=float)
    middle = np.mean(prices_arr)
    std = np.std(prices_arr, ddof=1)

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return (round(upper, 4), round(middle, 4), round(lower, 4))


def compute_macd(
    prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> dict:
    """
    Compute MACD (Moving Average Convergence/Divergence).

    Returns:
        {
            "macd_line": float,
            "signal_line": float,
            "histogram": float,
            "trend": "bullish" | "bearish" | "neutral"
        }
    """
    if len(prices) < slow + signal:
        return {"macd_line": 0, "signal_line": 0, "histogram": 0, "trend": "neutral"}

    prices_arr = np.array(prices, dtype=float)

    # EMA calculations
    ema_fast = _ema(prices_arr, fast)
    ema_slow = _ema(prices_arr, slow)

    # MACD line = fast EMA - slow EMA
    macd_line_arr = ema_fast[slow - fast:] - ema_slow
    if len(macd_line_arr) < signal:
        return {"macd_line": 0, "signal_line": 0, "histogram": 0, "trend": "neutral"}

    # Signal line = EMA of MACD line
    signal_line_arr = _ema(macd_line_arr, signal)

    macd_val = float(macd_line_arr[-1])
    signal_val = float(signal_line_arr[-1])
    histogram = macd_val - signal_val

    # Trend: bullish crossover, bearish crossover
    if len(macd_line_arr) >= 2 and len(signal_line_arr) >= 2:
        prev_diff = float(macd_line_arr[-2]) - float(signal_line_arr[-2])
        curr_diff = histogram
        if prev_diff < 0 and curr_diff > 0:
            trend = "bullish_crossover"
        elif prev_diff > 0 and curr_diff < 0:
            trend = "bearish_crossover"
        elif curr_diff > 0:
            trend = "bullish"
        else:
            trend = "bearish"
    else:
        trend = "neutral"

    return {
        "macd_line": round(macd_val, 4),
        "signal_line": round(signal_val, 4),
        "histogram": round(histogram, 4),
        "trend": trend,
    }


def compute_ema_crossover(prices: list[float], fast: int = 9, slow: int = 21) -> dict:
    """
    EMA Crossover signal (9/21 day by default).

    Returns:
        {
            "fast_ema": float,
            "slow_ema": float,
            "signal": "golden_cross" | "death_cross" | "above" | "below"
        }
    """
    if len(prices) < slow + 2:
        return {"fast_ema": 0, "slow_ema": 0, "signal": "neutral"}

    prices_arr = np.array(prices, dtype=float)
    ema_f = _ema(prices_arr, fast)
    ema_s = _ema(prices_arr, slow)

    # Align arrays
    min_len = min(len(ema_f), len(ema_s))
    ema_f = ema_f[-min_len:]
    ema_s = ema_s[-min_len:]

    fast_val = float(ema_f[-1])
    slow_val = float(ema_s[-1])

    if len(ema_f) >= 2:
        prev_diff = float(ema_f[-2]) - float(ema_s[-2])
        curr_diff = fast_val - slow_val
        if prev_diff < 0 and curr_diff > 0:
            signal = "golden_cross"
        elif prev_diff > 0 and curr_diff < 0:
            signal = "death_cross"
        elif curr_diff > 0:
            signal = "above"
        else:
            signal = "below"
    else:
        signal = "above" if fast_val > slow_val else "below"

    return {
        "fast_ema": round(fast_val, 4),
        "slow_ema": round(slow_val, 4),
        "signal": signal,
    }


def compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float:
    """
    Compute ATR (Average True Range) for volatility-adjusted stops.
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0.0

    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1])
        )
    )

    atr = np.mean(tr[-period:])
    return round(float(atr), 4)


def detect_volume_spike(volumes: list[float], lookback: int = 20, threshold: float = 2.0) -> dict:
    """
    Detect unusual volume spikes (institutional activity).

    Returns:
        {
            "current_volume": float,
            "avg_volume": float,
            "ratio": float,
            "is_spike": bool
        }
    """
    if len(volumes) < lookback + 1:
        return {"current_volume": 0, "avg_volume": 0, "ratio": 1.0, "is_spike": False}

    current = float(volumes[-1])
    avg = float(np.mean(volumes[-lookback - 1:-1]))

    if avg <= 0:
        return {"current_volume": current, "avg_volume": 0, "ratio": 0, "is_spike": False}

    ratio = current / avg
    return {
        "current_volume": current,
        "avg_volume": round(avg, 0),
        "ratio": round(ratio, 2),
        "is_spike": ratio >= threshold,
    }


def compute_trend_score(prices: list[float], volumes: list[float] = None,
                        highs: list[float] = None, lows: list[float] = None) -> dict:
    """
    Compute a composite trend score (0-100) combining all indicators.
    This is the CORE analysis function for the strategy agent.

    Returns:
        {
            "score": 0-100 (>70 strong buy, >55 buy, 45-55 neutral, <45 sell, <30 strong sell),
            "rsi": float,
            "macd": dict,
            "ema_cross": dict,
            "bollinger_position": "above_upper" | "near_upper" | "middle" | "near_lower" | "below_lower",
            "volume_spike": bool,
            "atr": float,
            "recommendation": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
        }
    """
    score = 50  # Start neutral

    # RSI component (+/- 15)
    rsi = compute_rsi(prices)
    if rsi < 30:
        score += 15  # Oversold = bullish
    elif rsi < 40:
        score += 8
    elif rsi > 70:
        score -= 15  # Overbought = bearish
    elif rsi > 60:
        score -= 5

    # MACD component (+/- 15)
    macd = compute_macd(prices)
    if macd["trend"] == "bullish_crossover":
        score += 15
    elif macd["trend"] == "bullish":
        score += 8
    elif macd["trend"] == "bearish_crossover":
        score -= 15
    elif macd["trend"] == "bearish":
        score -= 8

    # EMA Crossover component (+/- 10)
    ema = compute_ema_crossover(prices)
    if ema["signal"] == "golden_cross":
        score += 10
    elif ema["signal"] == "above":
        score += 4
    elif ema["signal"] == "death_cross":
        score -= 10
    elif ema["signal"] == "below":
        score -= 4

    # Bollinger position (+/- 10)
    upper, middle, lower = compute_bollinger_bands(prices)
    current_price = prices[-1] if prices else 0
    bb_range = upper - lower if upper != lower else 1

    if current_price <= lower:
        bb_pos = "below_lower"
        score += 10  # Oversold
    elif current_price <= lower + 0.2 * bb_range:
        bb_pos = "near_lower"
        score += 5
    elif current_price >= upper:
        bb_pos = "above_upper"
        score -= 10  # Overbought
    elif current_price >= upper - 0.2 * bb_range:
        bb_pos = "near_upper"
        score -= 5
    else:
        bb_pos = "middle"

    # Volume component (+/- 5)
    vol_spike = False
    if volumes and len(volumes) > 20:
        vol_data = detect_volume_spike(volumes)
        vol_spike = vol_data["is_spike"]
        if vol_spike and score > 50:
            score += 5  # Volume confirms bullish move
        elif vol_spike and score < 50:
            score += 5  # Volume on dip = potential reversal

    # ATR
    atr = 0.0
    if highs and lows:
        atr = compute_atr(highs, lows, prices)

    # Clamp score
    score = max(0, min(100, score))

    # Recommendation
    if score >= 70:
        rec = "strong_buy"
    elif score >= 55:
        rec = "buy"
    elif score >= 45:
        rec = "hold"
    elif score >= 30:
        rec = "sell"
    else:
        rec = "strong_sell"

    return {
        "score": score,
        "rsi": rsi,
        "macd": macd,
        "ema_cross": ema,
        "bollinger_position": bb_pos,
        "bollinger_bands": {"upper": upper, "middle": middle, "lower": lower},
        "volume_spike": vol_spike,
        "atr": atr,
        "recommendation": rec,
    }


def compute_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.05) -> float:
    """Compute annualized Sharpe Ratio."""
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns, dtype=float)
    mean_return = np.mean(returns_arr) * 252
    std_return = np.std(returns_arr, ddof=1) * np.sqrt(252)

    if std_return < 1e-10:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return round(sharpe, 4)


def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """Compute Kelly Criterion fraction (capped at 15% for equal-weight mode)."""
    if win_prob <= 0 or win_prob >= 1 or win_loss_ratio <= 0:
        return 0.0

    q = 1.0 - win_prob
    kelly = (win_loss_ratio * win_prob - q) / win_loss_ratio
    kelly = max(0.0, min(kelly, 0.15))  # Cap at 15% per position
    return round(kelly, 4)


def compute_daily_returns(prices: list[float]) -> list[float]:
    """Convert prices to daily returns."""
    if len(prices) < 2:
        return []
    prices_arr = np.array(prices, dtype=float)
    returns = np.diff(prices_arr) / prices_arr[:-1]
    return returns.tolist()


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average."""
    if len(data) < period:
        return data.copy()

    alpha = 2.0 / (period + 1)
    ema = np.zeros(len(data))
    ema[period - 1] = np.mean(data[:period])

    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema[period - 1:]


# ============================================
# PRO DAY-TRADING STRATEGIES
# ============================================


def compute_ema_trend_bias(prices: list[float], period: int = 100) -> dict:
    """
    Strategy 1: High-timeframe trend bias using 100 EMA.
    "Use 100 EMA on the 1H. Trade only in that direction."

    Returns:
        {"ema_100": float, "price_vs_ema": "above"|"below",
         "trend_bias": "bullish"|"bearish", "distance_pct": float}
    """
    if len(prices) < period:
        return {"ema_100": 0, "price_vs_ema": "unknown", "trend_bias": "neutral", "distance_pct": 0}

    arr = np.array(prices, dtype=float)
    ema_vals = _ema(arr, period)

    if len(ema_vals) == 0:
        return {"ema_100": 0, "price_vs_ema": "unknown", "trend_bias": "neutral", "distance_pct": 0}

    current_price = arr[-1]
    ema_current = ema_vals[-1]
    distance_pct = ((current_price - ema_current) / ema_current) * 100

    return {
        "ema_100": round(ema_current, 2),
        "price_vs_ema": "above" if current_price > ema_current else "below",
        "trend_bias": "bullish" if current_price > ema_current else "bearish",
        "distance_pct": round(distance_pct, 2),
    }


def compute_vwap(prices: list[float], volumes: list[float],
                 highs: list[float] = None, lows: list[float] = None) -> dict:
    """
    VWAP (Volume Weighted Average Price) + smart bounce detection.
    "Don't buy the first touch. Wait for a long wick, high volume, no follow-through."

    Returns:
        {"vwap": float, "price_vs_vwap": "above"|"below",
         "distance_pct": float, "smart_bounce": bool}
    """
    if not prices or not volumes or len(prices) != len(volumes):
        return {"vwap": 0, "price_vs_vwap": "unknown", "distance_pct": 0, "smart_bounce": False}

    p = np.array(prices, dtype=float)
    v = np.array(volumes, dtype=float)

    # Use typical price if highs/lows available
    if highs and lows and len(highs) == len(prices):
        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        typical = (h + l + p) / 3.0
    else:
        typical = p

    cumvol = np.cumsum(v)
    cumvol_price = np.cumsum(typical * v)

    # Avoid division by zero
    mask = cumvol > 0
    vwap_arr = np.where(mask, cumvol_price / cumvol, 0)
    vwap_current = vwap_arr[-1] if len(vwap_arr) > 0 else 0

    current_price = p[-1]
    distance_pct = ((current_price - vwap_current) / vwap_current * 100) if vwap_current else 0

    # Smart bounce detection:
    # Price near VWAP (within 0.5%), high volume on last bar, long wick (high-low > 2x body)
    smart_bounce = False
    if highs and lows and len(highs) >= 2 and abs(distance_pct) < 0.5:
        last_body = abs(p[-1] - p[-2])
        last_range = float(highs[-1]) - float(lows[-1])
        vol_ratio = v[-1] / np.mean(v[-20:]) if len(v) >= 20 else 1.0
        if last_range > 0 and last_body > 0:
            wick_ratio = last_range / last_body
            smart_bounce = wick_ratio > 2.0 and vol_ratio > 1.5

    return {
        "vwap": round(vwap_current, 2),
        "price_vs_vwap": "above" if current_price > vwap_current else "below",
        "distance_pct": round(distance_pct, 2),
        "smart_bounce": smart_bounce,
    }


def detect_rsi_exhaustion(prices: list[float], period: int = 14) -> dict:
    """
    Strategy 2: RSI exhaustion (rubber band setup).
    "If RSI hits 90+ (or sub-10) and hasn't pulled back,
     first reversal candle = go."

    Returns:
        {"rsi": float, "exhaustion": "overbought_extreme"|"oversold_extreme"|"none",
         "reversal_candle": bool, "setup_active": bool}
    """
    rsi = compute_rsi(prices, period)

    result = {
        "rsi": rsi,
        "exhaustion": "none",
        "reversal_candle": False,
        "setup_active": False,
    }

    if len(prices) < 3:
        return result

    # Check for extreme RSI
    if rsi >= 90:
        result["exhaustion"] = "overbought_extreme"
        # Check for reversal candle (current close < previous close)
        if prices[-1] < prices[-2]:
            result["reversal_candle"] = True
            result["setup_active"] = True  # SHORT signal
    elif rsi <= 10:
        result["exhaustion"] = "oversold_extreme"
        # Check for reversal candle (current close > previous close)
        if prices[-1] > prices[-2]:
            result["reversal_candle"] = True
            result["setup_active"] = True  # LONG signal

    return result


def detect_parabolic_reversal(prices: list[float], min_streak: int = 5) -> dict:
    """
    Strategy 2: Broken parabolic short.
    "Stock printed 5+ straight green candles? First red engulfing = short."

    Returns:
        {"green_streak": int, "red_streak": int,
         "parabolic_long": bool, "parabolic_short": bool,
         "reversal_detected": bool, "direction": "short"|"long"|"none"}
    """
    if len(prices) < min_streak + 1:
        return {"green_streak": 0, "red_streak": 0, "parabolic_long": False,
                "parabolic_short": False, "reversal_detected": False, "direction": "none"}

    # Count consecutive green/red candles ending at [-2] (before the latest)
    green_streak = 0
    red_streak = 0

    for i in range(len(prices) - 2, 0, -1):
        if prices[i] > prices[i - 1]:
            if red_streak > 0:
                break
            green_streak += 1
        elif prices[i] < prices[i - 1]:
            if green_streak > 0:
                break
            red_streak += 1
        else:
            break

    # Check if latest candle reverses the streak
    latest_green = prices[-1] > prices[-2]
    latest_red = prices[-1] < prices[-2]

    parabolic_long = green_streak >= min_streak and latest_red
    parabolic_short = red_streak >= min_streak and latest_green

    return {
        "green_streak": green_streak,
        "red_streak": red_streak,
        "parabolic_long": parabolic_long,   # Was going up, now reversing DOWN → short
        "parabolic_short": parabolic_short,  # Was going down, now reversing UP → long
        "reversal_detected": parabolic_long or parabolic_short,
        "direction": "short" if parabolic_long else ("long" if parabolic_short else "none"),
    }


def detect_stop_hunt_zones(prices: list[float], highs: list[float],
                           lows: list[float]) -> dict:
    """
    Strategy 3: Stop-loss hunting reversal.
    "Where does retail hide stops? Previous day high/low.
     Market fakes past it, then reverses. Trade against dumb money."

    Returns:
        {"prev_high": float, "prev_low": float,
         "above_prev_high": bool, "below_prev_low": bool,
         "stop_hunt_long": bool, "stop_hunt_short": bool}
    """
    if len(prices) < 3 or len(highs) < 2 or len(lows) < 2:
        return {"prev_high": 0, "prev_low": 0, "above_prev_high": False,
                "below_prev_low": False, "stop_hunt_long": False, "stop_hunt_short": False}

    # Previous day = second-to-last bar
    prev_high = float(highs[-2])
    prev_low = float(lows[-2])
    current_price = prices[-1]
    current_high = float(highs[-1])
    current_low = float(lows[-1])

    above_prev_high = current_high > prev_high
    below_prev_low = current_low < prev_low

    # Stop hunt long: price dipped below prev low then recovered (wick below, close above)
    stop_hunt_long = below_prev_low and current_price > prev_low

    # Stop hunt short: price spiked above prev high then dropped (wick above, close below)
    stop_hunt_short = above_prev_high and current_price < prev_high

    return {
        "prev_high": round(prev_high, 2),
        "prev_low": round(prev_low, 2),
        "above_prev_high": above_prev_high,
        "below_prev_low": below_prev_low,
        "stop_hunt_long": stop_hunt_long,
        "stop_hunt_short": stop_hunt_short,
    }


def detect_mm_refill(prices: list[float], volumes: list[float],
                     lookback: int = 10) -> dict:
    """
    Strategy 3: Market maker refill zones.
    "Stock grinds slow, then sudden volume spike with no movement.
     Big player filling orders."

    Returns:
        {"refill_detected": bool, "volume_ratio": float,
         "price_move_pct": float, "direction_hint": "accumulation"|"distribution"|"none"}
    """
    if len(prices) < lookback or len(volumes) < lookback:
        return {"refill_detected": False, "volume_ratio": 0,
                "price_move_pct": 0, "direction_hint": "none"}

    p = np.array(prices[-lookback:], dtype=float)
    v = np.array(volumes[-lookback:], dtype=float)

    avg_vol = np.mean(v[:-1]) if len(v) > 1 else 1
    last_vol = v[-1]
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

    # Price barely moved despite volume spike
    price_range = (np.max(p) - np.min(p)) / np.mean(p) * 100
    last_move = abs(p[-1] - p[-2]) / p[-2] * 100 if len(p) >= 2 else 0

    # High volume + tiny price move = refill
    refill = vol_ratio > 2.5 and last_move < 0.3

    direction = "none"
    if refill:
        # If price ended slightly up → accumulation (bullish)
        # If price ended slightly down → distribution (bearish)
        direction = "accumulation" if p[-1] >= p[-2] else "distribution"

    return {
        "refill_detected": refill,
        "volume_ratio": round(vol_ratio, 2),
        "price_move_pct": round(last_move, 2),
        "direction_hint": direction,
    }


def detect_first_hour_trend(prices: list[float], first_n_bars: int = 6) -> dict:
    """
    Strategy 1: First-hour trend lock.
    "Whatever the stock does in the first 30-60 min, stick with it."
    (Approximated using first N daily bars as proxy for intraday first hour)

    Returns:
        {"first_hour_direction": "bullish"|"bearish"|"neutral",
         "first_hour_change_pct": float,
         "current_aligned": bool}
    """
    if len(prices) < first_n_bars + 2:
        return {"first_hour_direction": "neutral", "first_hour_change_pct": 0, "current_aligned": True}

    # First N bars establish the trend
    open_price = prices[0]
    first_hour_close = prices[first_n_bars - 1]
    change_pct = ((first_hour_close - open_price) / open_price) * 100

    if change_pct > 0.3:
        direction = "bullish"
    elif change_pct < -0.3:
        direction = "bearish"
    else:
        direction = "neutral"

    # Is current price aligned with first-hour trend?
    current = prices[-1]
    if direction == "bullish":
        aligned = current > first_hour_close
    elif direction == "bearish":
        aligned = current < first_hour_close
    else:
        aligned = True

    return {
        "first_hour_direction": direction,
        "first_hour_change_pct": round(change_pct, 2),
        "current_aligned": aligned,
    }

