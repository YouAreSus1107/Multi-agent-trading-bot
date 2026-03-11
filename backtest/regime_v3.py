import pandas as pd
import numpy as np
import warnings
from hmmlearn.hmm import GaussianHMM

class MultiFactorRegimeModel:
    """
    V3 Regime Detection Model.
    
    Combines Trend, Volatility, and Chop/Efficiency scores to classify the broader market
    into 4 distinct regimes: bull_trend, bear_trend, chop, high_volatility_stress.
    
    Includes hysteresis (minimum dwell time) to prevent noisy whipsaw flips.
    """
    
    def __init__(self, 
                 volatility_threshold=0.22, 
                 atr_expansion_threshold=1.4,
                 chop_er_threshold=0.20,
                 chop_index_threshold=61.8,
                 trend_upper_threshold=30,
                 trend_lower_threshold=-30,
                 min_dwell_days=5):
        """
        Args:
            volatility_threshold: 25% annualized 20d volatility triggers stress
            atr_expansion_threshold: ATR(14) > 1.5x ATR(50) triggers stress
            chop_er_threshold: Kaufman's Efficiency Ratio < 0.20 indicates chop
            chop_index_threshold: Choppiness Index > 61.8 indicates chop
            trend_upper_threshold: Trend score needed to classify as bull
            trend_lower_threshold: Trend score needed to classify as bear
            min_dwell_days: Number of days a raw regime must persist before official switch
        """
        self.vol_threshold = volatility_threshold
        self.atr_expansion = atr_expansion_threshold
        self.er_threshold = chop_er_threshold
        self.chop_idx_threshold = chop_index_threshold
        self.trend_upper = trend_upper_threshold
        self.trend_lower = trend_lower_threshold
        self.dwell_days = min_dwell_days

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all orthogonal market features (Trend, Vol, Chop)."""
        df = df.copy()
        
        # Helper metrics
        df['ret'] = df['close'].pct_change()
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # --- 1. Volatility/Stress Features ---
        df['realized_vol_20d'] = df['ret'].rolling(20).std() * np.sqrt(252)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_50'] = df['tr'].rolling(50).mean()
        df['atr_ratio'] = df['atr_14'] / df['atr_50'].replace(0, np.nan)
        
        # --- 2. Chop/Efficiency Features ---
        # Kaufman Efficiency Ratio (14d)
        net_change = np.abs(df['close'] - df['close'].shift(14))
        sum_abs_changes = np.abs(df['close'].diff()).rolling(14).sum()
        df['er_14'] = net_change / sum_abs_changes.replace(0, np.nan)
        
        # Choppiness Index (14d)
        sum_tr_14 = df['tr'].rolling(14).sum()
        hh_14 = df['high'].rolling(14).max()
        ll_14 = df['low'].rolling(14).min()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # CHOP = 100 * LOG10( SUM(ATR(1), n) / ( HighestHigh(n) - LowestLow(n) ) ) / LOG10(n)
            chop_calc = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
            df['chop_14'] = chop_calc
        
        # --- 3. Trend Score Features ---
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Moving Average Alignment Score (-100 to +100)
        # 100 = Perfect bull (C > 20 > 50 > 200)
        # -100 = Perfect bear (C < 20 < 50 < 200)
        df['trend_score'] = 0
        
        # Bullish alignment logic
        df.loc[(df['close'] > df['ema_20']) & (df['ema_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200']), 'trend_score'] += 100
        df.loc[(df['close'] > df['ema_20']) & (df['ema_20'] > df['sma_50']) & (df['sma_50'] <= df['sma_200']), 'trend_score'] += 50
        df.loc[(df['close'] > df['ema_20']) & (df['ema_20'] <= df['sma_50']), 'trend_score'] += 20
        
        # Bearish alignment logic
        df.loc[(df['close'] < df['ema_20']) & (df['ema_20'] < df['sma_50']) & (df['sma_50'] < df['sma_200']), 'trend_score'] -= 100
        df.loc[(df['close'] < df['ema_20']) & (df['ema_20'] < df['sma_50']) & (df['sma_50'] >= df['sma_200']), 'trend_score'] -= 50
        df.loc[(df['close'] < df['ema_20']) & (df['ema_20'] >= df['sma_50']), 'trend_score'] -= 20
        
        # Add slight momentum bias to break ties or smooth score
        roc_20 = df['close'].pct_change(20)
        roc_score = np.clip(roc_20 * 10, -1, 1) * 20  # Max +/- 20 points from ROC
        df['trend_score'] = np.clip(df['trend_score'] + roc_score, -100, 100)
        
        # Smooth trend score slightly to prevent single-day jolts
        df['trend_score'] = df['trend_score'].rolling(3).mean()
        
        return df

    def _apply_state_machine(self, df: pd.DataFrame) -> pd.Series:
        """Maps orthogonal scores to a raw daily regime label."""
        raw_regime = pd.Series(index=df.index, dtype='object')
        raw_regime.fillna('chop', inplace=True) # Default state
        
        for idx, row in df.iterrows():
            if pd.isna(row['close']):
                continue
                
            # Rule 1: High Volatility Stress (Circuit Breaker)
            # If vol is too high or ATR is exploding, override trend.
            is_stress = False
            if pd.notna(row['realized_vol_20d']) and row['realized_vol_20d'] > self.vol_threshold:
                is_stress = True
            if pd.notna(row['atr_ratio']) and row['atr_ratio'] > self.atr_expansion:
                is_stress = True
                
            if is_stress:
                raw_regime[idx] = 'high_volatility_stress'
                continue
                
            # Rule 2: Chop / Low Efficiency
            # If market is moving sideways inefficiently
            is_chop = False
            if pd.notna(row['er_14']) and row['er_14'] < self.er_threshold:
                is_chop = True
            if pd.notna(row['chop_14']) and row['chop_14'] > self.chop_idx_threshold:
                is_chop = True
                
            if is_chop:
                raw_regime[idx] = 'chop'
                continue
                
            # Rule 3: Trend
            ts = row.get('trend_score', 0)
            if pd.isna(ts):
                ts = 0
                
            if ts >= self.trend_upper:
                raw_regime[idx] = 'bull_trend'
            elif ts <= self.trend_lower:
                raw_regime[idx] = 'bear_trend'
            else:
                raw_regime[idx] = 'chop'
                
        return raw_regime

    def _apply_hysteresis(self, raw_regimes: pd.Series) -> pd.Series:
        """
        Applies a minimum dwell time anti-whipsaw filter.
        A state must persist for 'dwell_days' before becoming the official regime.
        Exception: high_volatility_stress triggers instantly.
        """
        smoothed = pd.Series(index=raw_regimes.index, dtype='object')
        
        current_official_state = 'chop'
        candidate_state = 'chop'
        candidate_days = 0
        
        for i in range(len(raw_regimes)):
            day_state = raw_regimes.iloc[i]
            
            if pd.isna(day_state):
                smoothed.iloc[i] = current_official_state
                continue
                
            # Instant override for stress circuit-breaker
            if day_state == 'high_volatility_stress':
                current_official_state = 'high_volatility_stress'
                candidate_state = day_state
                candidate_days = 1
                smoothed.iloc[i] = current_official_state
                continue
                
            # If same state as candidate, increment counter
            if day_state == candidate_state:
                candidate_days += 1
            else:
                # Reset candidate
                candidate_state = day_state
                candidate_days = 1
                
            # If candidate persists long enough, it becomes official
            if candidate_days >= self.dwell_days:
                current_official_state = candidate_state
                
            smoothed.iloc[i] = current_official_state
            
        return smoothed

    def compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point.
        Expects a DataFrame with ['open', 'high', 'low', 'close', 'volume'].
        Returns the original DataFrame with new columns:
        - trend_score
        - realized_vol_20d
        - er_14 (efficiency ratio)
        - chop_14 (choppiness index)
        - regime_raw
        - regime (the final smoothed state)
        """
        # Ensure we have required columns
        req_cols = ['high', 'low', 'close']
        for c in req_cols:
            if c not in df.columns:
                raise ValueError(f"Input DataFrame must contain '{c}' column.")
                
        # 1. Compute Orthogonal Features
        featured_df = self._compute_features(df)
        
        # 2. Apply Rule-Based State Machine Mapping
        raw_regimes = self._apply_state_machine(featured_df)
        featured_df['regime_raw'] = raw_regimes
        
        # 3. Apply Anti-Whipsaw Filter
        smoothed_regimes = self._apply_hysteresis(raw_regimes)
        featured_df['regime'] = smoothed_regimes
        
        return featured_df

class HMMRegimeModel:
    """Unsupervised Machine Learning for Bull/Bear Regime Detection"""
    def __init__(self, n_regimes=2):
        self.n_regimes = n_regimes

    def compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        train_df = df[['returns', 'volatility']].dropna()
        X = np.column_stack([train_df['returns'], train_df['volatility']])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(n_components=self.n_regimes, covariance_type="full", n_iter=1000, random_state=42)
            model.fit(X)
            hidden_states = model.predict(X)
            
        # Dynamically identify Bear state (always the state with higher variance)
        state_variances = np.diag(model.covars_[:, 1, 1]) 
        bear_state = 0 if state_variances[0] > state_variances[1] else 1
        bull_state = 1 if bear_state == 0 else 0
            
        state_map = {bull_state: 'bull', bear_state: 'bear'}
        df['regime'] = 'neutral'
        df.loc[train_df.index, 'regime'] = [state_map[state] for state in hidden_states]
        df['regime'] = df['regime'].replace('neutral', method='bfill')
        
        return df[['close', 'regime']]
