import os
import pandas as pd
import numpy as np
import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

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


# NOTE: The first HMMRegimeModel definition (5-feature curated version) has been
# removed. The DataDave1 methodology class below (27-feature, BIC-optimal) is the
# sole implementation. It was being silently shadowed by the second definition anyway.


class HMMRegimeModel:
    """
    Article-faithful HMM Regime Detector (DataDave1 / Medium methodology).

    Pipeline:
      1. Fetch auxiliary data (VIX, VIX3M, HYG, LQD) via yfinance, disk-cached.
      2. Build 27 cross-asset features: multi-horizon returns, realized vol,
         VIX term structure, credit spread, drawdowns, MA alignment flags.
      3. StandardScaler + PCA (95% variance) for stable uncorrelated input.
      4. BIC-optimal GaussianHMM for K in 3..9, 5 random seeds each.
      5. Viterbi decode + diagnostics table -> label states by ann_ret_%.
      6. filtered_probability() for real-time regime probs (live trading use).
    """

    CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, k_range: range = range(3, 10), n_seeds: int = 5,
                 variance_target: float = 0.95, min_dwell_days: int = 5):
        self.k_range = k_range
        self.n_seeds = n_seeds
        self.variance_target = variance_target
        self.min_dwell_days = min_dwell_days
        # Set after fitting
        self._hmm = None
        self._scaler = None
        self._pca = None
        self._n_pca = None
        self._diagnostics = None
        self._state_map = {}

    # ------------------------------------------------------------------ #
    # Step 1: Aux data (VIX, VIX3M, HYG, LQD) via yfinance, disk cached  #
    # ------------------------------------------------------------------ #
    def _fetch_aux(self, start: str = "2010-01-01") -> pd.DataFrame:
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(self.CACHE_DIR, "hmm_aux_data.parquet")

        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if not cached.empty and (pd.Timestamp.now() - cached.index[-1]).days < 5:
                print("  [HMM] Loaded aux data (VIX/RUT/HYG/LQD) from cache.")
                return cached

        # Full symbol set matching article: VIX, VIX3M, VIX6M, RUT, HYG, LQD
        print("  [HMM] Fetching VIX, VIX3M, VIX6M, ^RUT, HYG, LQD via yfinance...")
        try:
            import yfinance as yf
            symbols = {
                "^VIX":  "vix",
                "^VIX3M": "vix3m",
                "^VIX6M": "vix6m",
                "^RUT":  "rut",
                "HYG":   "HYG",
                "LQD":   "LQD",
            }
            frames = {}
            for sym, name in symbols.items():
                try:
                    raw = yf.download(sym, start=start, progress=False, auto_adjust=True)
                    if not raw.empty:
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = raw.columns.get_level_values(0)
                        raw.index = pd.to_datetime(raw.index)
                        if raw.index.tz is not None:
                            raw.index = raw.index.tz_localize(None)
                        frames[name] = raw["Close"].rename(name)
                        print(f"    {sym}: {len(raw)} rows")
                except Exception as e:
                    print(f"  [HMM] Warning: could not fetch {sym}: {e}")

            if frames:
                # Delete stale cache so it gets refreshed
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                aux = pd.concat(frames.values(), axis=1).sort_index()
                aux.to_parquet(cache_file)
                print(f"  [HMM] Cached {len(aux)} rows of aux data ({len(frames)} symbols).")
                return aux
        except ImportError:
            print("  [HMM] yfinance unavailable -- VIX/credit features will be skipped.")
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Step 2: Feature engineering                                          #
    # ------------------------------------------------------------------ #
    def _build_features(self, snp_df: pd.DataFrame,
                        aux_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Article Step 2 feature engineering -- exact match.
        SPX features + RUT cross-asset features + VIX term structure + credit spread.
        """
        close = snp_df["close"].copy()
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        out = pd.DataFrame(index=close.index)

        # === SPX features ===
        # MA alignment flags (breadth)
        out["SPX_Above50D"]  = (close > close.rolling(50).mean()).astype(float)
        out["SPX_Above200D"] = (close > close.rolling(200).mean()).astype(float)

        # Multi-horizon returns
        out["SPX_Daily_Return"] = close.pct_change()
        out["SPX_21D_Return"]   = close.pct_change(21)
        out["SPX_63D_Return"]   = close.pct_change(63)
        out["SPX_126D_Return"]  = close.pct_change(126)

        # Realized volatility (annualised)
        spx_dr = out["SPX_Daily_Return"]
        out["SPX_21D_RealVol"] = spx_dr.rolling(21).std() * np.sqrt(252)
        out["SPX_63D_RealVol"] = spx_dr.rolling(63).std() * np.sqrt(252)

        # 6-month drawdown
        spx_roll_max = close.rolling(126, min_periods=1).max()
        out["SPX_126D_Drawdown"] = close / spx_roll_max - 1.0

        # === Auxiliary data: RUT, VIX, HYG/LQD ===
        if aux_df is not None and not aux_df.empty:
            aux = aux_df.copy()
            aux.index = pd.to_datetime(aux.index)
            if aux.index.tz is not None:
                aux.index = aux.index.tz_localize(None)
            aux = aux.reindex(out.index, method="ffill")

            # --- RUT: multi-horizon returns, realized vol, drawdown ---
            if "rut" in aux.columns:
                rut = aux["rut"]
                rut_dr = rut.pct_change()
                out["RUT_Daily_Return"] = rut_dr
                out["RUT_21D_Return"]   = rut.pct_change(21)
                out["RUT_63D_Return"]   = rut.pct_change(63)
                out["RUT_126D_Return"]  = rut.pct_change(126)
                out["RUT_21D_RealVol"]  = rut_dr.rolling(21).std() * np.sqrt(252)
                out["RUT_63D_RealVol"]  = rut_dr.rolling(63).std() * np.sqrt(252)
                rut_roll_max = rut.rolling(126, min_periods=1).max()
                out["RUT_126D_Drawdown"] = rut / rut_roll_max - 1.0

                # SPX - RUT style tilt (large vs small cap relative performance)
                out["SPX_vs_RUT_21D"] = out["SPX_21D_Return"] - out["RUT_21D_Return"]
                out["SPX_vs_RUT_63D"] = out["SPX_63D_Return"] - out["RUT_63D_Return"]

                # Cross-asset 63D correlation and RUT beta to SPX
                out["SPX_RUT_63D_Corr"] = spx_dr.rolling(63).corr(rut_dr)
                cov = spx_dr.rolling(63).cov(rut_dr)
                var = spx_dr.rolling(63).var()
                out["RUT_beta_SPX"] = cov / var.replace(0, np.nan)

            # --- VIX: level, changes, term structure ratio, implied/realized spread ---
            if "vix" in aux.columns:
                out["VIX"]               = aux["vix"]
                out["VIX_1D_Change"]     = aux["vix"].diff(1)
                out["VIX_5D_Change"]     = aux["vix"].diff(5)
                out["VIX_to_SPXRealVol"] = aux["vix"] / out["SPX_21D_RealVol"].replace(0, np.nan)

            if "vix3m" in aux.columns and "vix" in aux.columns:
                out["VIX3M_VIX"] = aux["vix3m"] / aux["vix"].replace(0, np.nan)

            if "vix6m" in aux.columns and "vix" in aux.columns:
                out["VIX6M_VIX"] = aux["vix6m"] / aux["vix"].replace(0, np.nan)

            # --- Credit spread: log(HYG) - log(LQD) --rises in stress ---
            if "HYG" in aux.columns and "LQD" in aux.columns:
                out["Credit_Spread"] = (
                    np.log(aux["HYG"].replace(0, np.nan)) -
                    np.log(aux["LQD"].replace(0, np.nan))
                )

        return out

    # ------------------------------------------------------------------ #
    # Step 3: BIC model selection (article exact formula)                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bic(hmm_model: GaussianHMM, X: np.ndarray) -> tuple[float, float]:
        K = hmm_model.n_components
        T, d = X.shape
        logL = hmm_model.score(X)
        k_params = (K - 1) + (K * (K - 1)) + K * (2 * d)
        bic = -2.0 * logL + k_params * np.log(T)
        return bic, logL

    def _select_best_hmm(self, X: np.ndarray) -> GaussianHMM:
        best = {"bic": np.inf, "hmm": None, "K": None}
        print(f"  [HMM] BIC search K={list(self.k_range)}, {self.n_seeds} seeds each:")
        for K in self.k_range:
            best_k = {"bic": np.inf, "hmm": None}
            for seed in range(self.n_seeds):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        hmm = GaussianHMM(
                            n_components=K, covariance_type="diag",
                            n_iter=1000, random_state=42 + seed, verbose=False
                        )
                        hmm.fit(X)
                    bic, logL = self._bic(hmm, X)
                    if bic < best_k["bic"]:
                        best_k = {"bic": bic, "hmm": hmm}
                except Exception:
                    pass
            if best_k["hmm"] is not None:
                print(f"    K={K}: BIC={best_k['bic']:,.1f}")
                if best_k["bic"] < best["bic"]:
                    best = {"bic": best_k["bic"], "hmm": best_k["hmm"], "K": K}

        print(f"  [HMM] Selected K={best['K']} (BIC={best['bic']:,.1f})")
        return best["hmm"]

    # ------------------------------------------------------------------ #
    # Step 4: Diagnostics table (article Step 6)                          #
    # ------------------------------------------------------------------ #
    def _build_diagnostics(self, states: np.ndarray,
                           feat_df: pd.DataFrame) -> pd.DataFrame:
        K = self._hmm.n_components
        dr  = feat_df.get("SPX_Daily_Return",  pd.Series(np.zeros(len(states)), index=feat_df.index)).values
        vol = feat_df.get("SPX_21D_RealVol",   pd.Series(np.zeros(len(states)), index=feat_df.index)).values
        dd  = feat_df.get("SPX_126D_Drawdown", pd.Series(np.zeros(len(states)), index=feat_df.index)).values
        rows = []
        for s in range(K):
            mask = (states == s)
            rows.append({
                "state":          s,
                "obs_count":      int(mask.sum()),
                "mean_daily_ret": float(np.nanmean(dr[mask]))  if mask.any() else 0.0,
                "ann_ret_%":      float(np.nanmean(dr[mask]) * 252 * 100) if mask.any() else 0.0,
                "ann_vol_%":      float(np.nanmean(vol[mask]) * 100) if mask.any() else 0.0,
                "avg_drawdown_%": float(np.nanmean(dd[mask]) * 100)  if mask.any() else 0.0,
            })
        return pd.DataFrame(rows).sort_values(
            ["ann_ret_%", "mean_daily_ret"], ascending=[False, False]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Step 5: Dwell filter                                                 #
    # ------------------------------------------------------------------ #
    def _apply_dwell_filter(self, raw_labels: list) -> list:
        smoothed  = []
        current   = raw_labels[0]
        candidate = raw_labels[0]
        streak    = 1
        for label in raw_labels:
            if label == candidate:
                streak += 1
            else:
                candidate = label
                streak    = 1
            if streak >= self.min_dwell_days:
                current = candidate
            smoothed.append(current)
        return smoothed

    # ------------------------------------------------------------------ #
    # Step 6: Filtered forward probability (article Step 8 exact)         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logsumexp(arr: np.ndarray) -> float:
        m = np.max(arr)
        return m + np.log(np.exp(arr - m).sum())

    def _filtered_probs_all(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm in log-space. Returns probs mapping for EVERY observation (shape T x K)."""
        log_start  = np.log(np.clip(self._hmm.startprob_, 1e-300, None))
        log_trans  = np.log(np.clip(self._hmm.transmat_,  1e-300, None))
        log_b      = self._hmm._compute_log_likelihood(X)
        T, K       = log_b.shape
        log_alpha  = np.empty((T, K))
        log_alpha[0] = log_start + log_b[0]
        log_alpha[0] -= self._logsumexp(log_alpha[0])
        for t in range(1, T):
            trans        = log_alpha[t - 1][:, None] + log_trans
            log_alpha[t] = np.logaddexp.reduce(trans, axis=0) + log_b[t]
            log_alpha[t] -= self._logsumexp(log_alpha[t])
        
        alpha = np.exp(log_alpha)
        return alpha / alpha.sum(axis=1, keepdims=True)

    def _filtered_probs_last(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm in log-space. Returns probs for the LAST observation."""
        log_start = np.log(np.clip(self._hmm.startprob_, 1e-300, None))
        log_trans  = np.log(np.clip(self._hmm.transmat_,  1e-300, None))
        log_b      = self._hmm._compute_log_likelihood(X)
        T, K       = log_b.shape
        log_alpha  = np.empty((T, K))
        log_alpha[0] = log_start + log_b[0]
        log_alpha[0] -= self._logsumexp(log_alpha[0])
        for t in range(1, T):
            trans        = log_alpha[t - 1][:, None] + log_trans
            log_alpha[t] = np.logaddexp.reduce(trans, axis=0) + log_b[t]
            log_alpha[t] -= self._logsumexp(log_alpha[t])
        last = np.exp(log_alpha[-1])
        return last / last.sum()

    # ------------------------------------------------------------------ #
    # Step 7: Visualization (article exact color scheme)                   #
    # ------------------------------------------------------------------ #
    def plot_regimes(self, X_pca: np.ndarray, states: np.ndarray,
                     dates: pd.Index, prices: pd.Series = None, save_path: str = None) -> None:
        """
        Article Step 7: PCA scatter colored by regime + price timeline.
        Highest ann_ret% state = green (#2ca02c)
        Lowest ann_ret% state  = red   (#d62728)
        Middle states use fallback palette.
        Saves to save_path (HTML) if provided, otherwise shows interactively.
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("  [HMM] plotly not installed -- skipping visualization. pip install plotly")
            return

        if self._diagnostics is None:
            print("  [HMM] No diagnostics available -- run compute_regime() first.")
            return

        # Article color assignment: ranked by ann_ret% best->worst
        state_order = self._diagnostics["state"].tolist()
        fallback_colors = ["#8fd18f", "#ffbf86", "#ff8c42", "#9467bd", "#1f77b4", "#7f7f7f"]
        regime_colors = {}
        for i, s in enumerate(state_order):
            if i == 0:
                regime_colors[s] = "#2ca02c"   # best (green)
            elif i == len(state_order) - 1:
                regime_colors[s] = "#d62728"   # worst (red)
            else:
                regime_colors[s] = fallback_colors[(i - 1) % len(fallback_colors)]

        def state_label(s): return f"State {s}"
        
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("HMM Regime Clusters in PCA Space (PC1 vs PC2)", "^GSPC with Regime Coloring"),
            vertical_spacing=0.15
        )
        
        # Add scatter to row 1
        for s in state_order:
            mask = (states == s)
            if not mask.any(): continue
            lbl = self._state_map.get(s, "?")
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                name=f"Regime {s} ({lbl})",
                marker=dict(color=regime_colors[s], size=5, opacity=0.75),
                text=dates[mask] if hasattr(dates, 'values') else dates,
                hovertemplate="Date: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
            ), row=1, col=1)

        # Add price line to row 2
        if prices is not None:
            dates_arr = dates.values if hasattr(dates, 'values') else dates
            # Add thin gray line to connect gaps cleanly
            fig.add_trace(go.Scatter(
                x=dates_arr, y=prices,
                mode="lines",
                showlegend=False,
                line=dict(color="lightgray", width=1),
                hoverinfo="skip"
            ), row=2, col=1)
            
            # Overlay regime colors
            for s in state_order:
                mask = (states == s)
                if not mask.any(): continue
                fig.add_trace(go.Scatter(
                    x=dates_arr[mask], y=prices[mask],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=regime_colors[s], size=3),
                    hovertemplate="Date: %{x}<br>Price: %{y:.2f}<extra></extra>"
                ), row=2, col=1)

        fig.update_layout(height=1000, width=1100, title_text="HMM Regime Analysis", title_x=0.5)

        if save_path:
            html = fig.to_html()
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  [HMM] Regime charts saved to {save_path}")
        else:
            fig.show()

    def get_current_regime_probs(self, snp_df: pd.DataFrame,
                                 aux_df: pd.DataFrame = None) -> dict:
        """
        Live-safe: returns {'bull': 0.72, 'bear': 0.28} using forward algorithm.
        No future data used. Call this instead of compute_regime() in live trading.
        """
        if self._hmm is None:
            raise RuntimeError("Call compute_regime() first to fit the model.")
        feat  = self._build_features(snp_df, aux_df).dropna()
        X     = self._to_pca(feat.values.astype(float))
        probs = self._filtered_probs_last(X)
        result = {}
        for s, p in enumerate(probs):
            lbl = self._state_map.get(s, f"state_{s}")
            result[lbl] = result.get(lbl, 0.0) + float(p)
        return result

    # ------------------------------------------------------------------ #
    # Internal: scale + PCA transform                                      #
    # ------------------------------------------------------------------ #
    def _to_pca(self, X_raw: np.ndarray) -> np.ndarray:
        X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
        col_med = np.nanmedian(X_raw, axis=0)
        nans = np.where(np.isnan(X_raw))
        X_raw[nans] = np.take(col_med, nans[1])
        X_scaled = self._scaler.transform(X_raw)
        X_pca    = self._pca.transform(X_scaled)
        return X_pca[:, :self._n_pca]

    # ------------------------------------------------------------------ #
    # Main: compute_regime()                                               #
    # ------------------------------------------------------------------ #
    def compute_regime(self, snp_df: pd.DataFrame,
                       aux_df: pd.DataFrame = None,
                       fetch_aux: bool = True,
                       train_end: str = None,
                       save_chart: bool = True) -> pd.DataFrame:
        """
        Full pipeline. snp_df = OHLCV DataFrame (equal-weighted S&P or SPY).
        If train_end is given, Standard Scaling, PCA, and HMM are fit ONLY on data up to train_end.
        Inference on the full sequence is then done via real-time forward algorithm (NO look-ahead bias).
        Returns df[['close', 'regime']] indexed like snp_df.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = snp_df.copy()
        df.index = pd.to_datetime(df.index)

        # 1. Aux data
        if aux_df is None and fetch_aux:
            aux_df = self._fetch_aux()

        # 2. Features
        print("  [HMM] Engineering features...")
        feat_df  = self._build_features(df, aux_df)
        
        if train_end is not None:
            train_df = feat_df.loc[:train_end].dropna()
            print(f"  [HMM] Training history up to {train_end}: {len(train_df)} valid rows.")
        else:
            train_df = feat_df.dropna()
            print(f"  [HMM] {len(train_df.columns)} features, {len(train_df)} valid rows.")

        if len(train_df) < 50:
            df["regime"] = "bull"
            return df[["close", "regime"]]

        # 3. Clean & scale (FIT ON TRAIN ONLY)
        X_train_raw = train_df.values.astype(float)
        X_train_raw = np.where(np.isinf(X_train_raw), np.nan, X_train_raw)
        col_med = np.nanmedian(X_train_raw, axis=0)
        nans_t = np.where(np.isnan(X_train_raw))
        X_train_raw[nans_t] = np.take(col_med, nans_t[1])
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train_raw)

        # 4. PCA (FIT ON TRAIN ONLY)
        self._pca   = PCA()
        X_pca       = self._pca.fit_transform(X_scaled)
        evr         = self._pca.explained_variance_ratio_
        self._n_pca = int(np.argmax(np.cumsum(evr) >= self.variance_target) + 1)
        X_final     = X_pca[:, :self._n_pca]
        print(f"  [HMM] PCA: {self._n_pca} components -> "
              f"{np.cumsum(evr)[self._n_pca - 1]:.1%} variance explained.")

        # 5. BIC-optimal HMM (FIT ON TRAIN ONLY)
        self._hmm = self._select_best_hmm(X_final)

        # 6. Viterbi path (ONLY for state diagnostics/labels calculation based on train data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_states = self._hmm.predict(X_final)

        # 7. Diagnostics + label states
        # Strategy: map ann_ret < -10% -> bear, > 10% -> bull, else -> chop
        self._diagnostics = self._build_diagnostics(raw_states, train_df)
        self._state_map   = {}
        for _, row in self._diagnostics.iterrows():
            s_id = int(row["state"])
            ann_ret = row["ann_ret_%"]
            if ann_ret < -10:
                self._state_map[s_id] = "bear"
            elif ann_ret > 10:
                self._state_map[s_id] = "bull"
            else:
                self._state_map[s_id] = "chop"

        print("\n  [HMM] State diagnostics (best -> worst):")
        print(f"  {'St':<4} {'Days':<6} {'AnnRet%':<10} {'AnnVol%':<10} "
              f"{'AvgDD%':<10} {'Label'}")
        print(f"  {'-'*50}")
        for _, row in self._diagnostics.iterrows():
            lbl = self._state_map.get(int(row['state']), '?')
            print(f"  {int(row['state']):<4} {int(row['obs_count']):<6} "
                  f"{row['ann_ret_%']:<10.2f} {row['ann_vol_%']:<10.2f} "
                  f"{row['avg_drawdown_%']:<10.2f} {lbl}")

        # 8. Out-of-Sample Inference (Transform FULL dataset, use strictly causal forward-algorithm)
        full_valid_df = feat_df.dropna()
        X_full_raw = full_valid_df.values.astype(float)
        X_full_raw = np.where(np.isinf(X_full_raw), np.nan, X_full_raw)
        
        # Fill incoming full dataset NaNs with TRAIN medians
        nans_full = np.where(np.isnan(X_full_raw))
        X_full_raw[nans_full] = np.take(col_med, nans_full[1])
        
        X_full_scaled = self._scaler.transform(X_full_raw)
        X_full_pca = self._pca.transform(X_full_scaled)[:, :self._n_pca]
        
        # Calculate purely filtered (causal) probability for every time step
        probs_all = self._filtered_probs_all(X_full_pca)
        raw_states_all = np.argmax(probs_all, axis=1)

        # 9. Real-time Dwell filter
        raw_labels = [self._state_map.get(int(s)) for s in raw_states_all]
        smoothed_labels = self._apply_dwell_filter(raw_labels)

        # Re-align with full DataFrame index
        df["regime"] = pd.Series(smoothed_labels, index=full_valid_df.index)
        df["regime"] = df["regime"].ffill().fillna("bull")

        # 10. Visualization (Step 7) -- save PCA scatter to HTML next to report
        if save_chart:
            try:
                out_dir = os.path.join(os.path.dirname(__file__), "outputs")
                os.makedirs(out_dir, exist_ok=True)
                chart_path = os.path.join(out_dir, "regime_pca_chart.html")
                prices_for_chart = df.loc[full_valid_df.index, "close"]
                self.plot_regimes(X_full_pca, raw_states_all, full_valid_df.index, prices=prices_for_chart, save_path=chart_path)
            except Exception as e:
                import sys
                sys.stdout.buffer.write(f"  [HMM] Visualization skipped: {e}\n".encode("utf-8", errors="replace"))

        return df[["close", "regime"]]

