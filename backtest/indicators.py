import pandas as pd
import numpy as np

def compute_wma(series: pd.Series, period: int) -> pd.Series:
    """
    Computes the Weighted Moving Average (WMA) for a given pandas Series.
    wma = sum(price * weight) / sum(weights)
    """
    if len(series) < period:
        return pd.Series(index=series.index, dtype=float)
    
    weights = np.arange(1, period + 1)
    
    def wma_calc(x):
        return np.dot(x, weights) / weights.sum()
        
    return series.rolling(window=period).apply(wma_calc, raw=True)

def compute_hma(series: pd.Series, period: int) -> pd.Series:
    """
    Standard Hull Moving Average (HMA).
    hma = wma(2 * wma(x, p/2) - wma(x, p), sqrt(p))
    """
    half_length = max(1, period // 2)
    sqrt_length = max(1, int(np.round(np.sqrt(period))))

    wma_half = compute_wma(series, half_length)
    wma_full = compute_wma(series, period)

    raw_hma = 2 * wma_half - wma_full
    return compute_wma(raw_hma, sqrt_length)

def compute_hma3(series: pd.Series, period: int) -> pd.Series:
    """
    Triple-weighted HMA variant (HMA3).
    hma3 = wma(wma(x, p/3)*3 - wma(x, p/2) - wma(x, p), p)
    """
    third_length = max(1, period // 3)
    half_length = max(1, period // 2)

    wma_third = compute_wma(series, third_length)
    wma_half = compute_wma(series, half_length)
    wma_full = compute_wma(series, period)

    raw_hma3 = (wma_third * 3) - wma_half - wma_full
    return compute_wma(raw_hma3, period)

def apply_kalman_filter(series: pd.Series, gain: float = 0.7) -> pd.Series:
    """
    Recursive Kalman smoothing matching the Pine Script implementation:
    prevKf = nz(kf[1], x)
    dk = x - prevKf
    smooth = prevKf + dk * sqrt(gain * 2)
    velo = nz(velo[1], 0.0) + gain * dk
    kf = smooth + velo
    """
    kf_val = np.nan
    velo = 0.0
    
    result = np.full(len(series), np.nan)
    sqrt_gain2 = np.sqrt(gain * 2)

    # Convert to array for faster element-wise iteration
    arr = series.values
    
    for i in range(len(arr)):
        x = arr[i]
        
        # PineScript `nz()` handles NaNs
        if np.isnan(x):
            continue
            
        if np.isnan(kf_val):
            # First valid point: initialize prevKf to x
            prevKf = x
            velo = 0.0
        else:
            prevKf = kf_val
            
        dk = x - prevKf
        smooth = prevKf + dk * sqrt_gain2
        velo = velo + gain * dk
        kf_val = smooth + velo
        
        result[i] = kf_val

    return pd.Series(result, index=series.index)


def compute_hma_kahlman_regime(
    sp500_daily_closes: pd.Series, 
    hk_length: int = 14, 
    kalman_gain: float = 0.7,
    buffer_days: int = 2
) -> pd.DataFrame:
    """
    Computes the full HMA-Kahlman Trend and extracts the broad market Regime (Bull/Bear).
    
    Args:
        sp500_daily_closes: pd.Series of daily S&P 500 closing prices.
        hk_length: Length for HMA (default 14).
        kalman_gain: Responsiveness gain for Kalman filter (default 0.7).
        buffer_days: Number of consecutive days line_b must be > line_a to declare Bull regime (prevents whipsaw).
        
    Returns:
        pd.DataFrame containing `line_a`, `line_b`, and `regime` ('bull' or 'bear').
    """
    # 1. Base HMAs
    hma_val = compute_hma(sp500_daily_closes, hk_length)
    hma3_val = compute_hma3(sp500_daily_closes, max(2, hk_length // 2))

    # 2. Kalman Filter Smoothing
    line_a = apply_kalman_filter(hma_val, kalman_gain)
    line_b = apply_kalman_filter(hma3_val, kalman_gain)
    
    # 3. Regime Calculation (Line B > Line A = Bullish)
    # Using a buffer to ensure the crossover holds for N periods to prevent chop.
    is_bull_raw = line_b > line_a
    
    df = pd.DataFrame({
        'close': sp500_daily_closes,
        'line_a': line_a,
        'line_b': line_b,
        'is_bull_raw': is_bull_raw
    })
    
    # Apply Buffer (Must be solidly true or false for N days to change state)
    df['regime'] = 'neutral'
    
    # Rolling sum to detect N consecutive days of bull/bear
    rolling_bull = df['is_bull_raw'].rolling(window=buffer_days).sum()
    
    current_regime = 'bear' # Default start state
    regimes = []
    
    for i in range(len(df)):
        if pd.isna(rolling_bull.iloc[i]):
            regimes.append(current_regime)
            continue
            
        # If it's been bullish for exactly the buffer days, switch to bull
        if rolling_bull.iloc[i] == buffer_days:
            current_regime = 'bull'
        # If it's been bearish (0 bullish days) for exactly the buffer days, switch to bear
        elif rolling_bull.iloc[i] == 0:
            current_regime = 'bear'
        
        regimes.append(current_regime)
        
    df['regime'] = regimes
    
    return df[['line_a', 'line_b', 'regime']]
