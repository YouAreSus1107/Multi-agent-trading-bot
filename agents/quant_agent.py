"""
Quant Analyst Agent
Mathematical engine for Factor Attribution, XGBoost Probability, GBM Risk Modeling,
and Full-Market Residual Correlation Scanning.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import norm
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import json
import os

from utils.logger import get_logger

logger = get_logger("quant_analyst")

# Suppress warnings from yfinance and statsmodels
warnings.filterwarnings('ignore')

UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "market_universe.json")

# Minimum thresholds for scanner candidates
MIN_PRICE = 3.0             # No penny stocks
MIN_AVG_VOLUME = 500_000    # Minimum average daily volume (shares)
RESIDUAL_LOOKBACK = 5       # Days of residual momentum to sum
TOP_N_CANDIDATES = 30       # How many tickers to return


class QuantAnalystAgent:
    """Pre-market quantitative mathematical modeling."""

    def __init__(self):
        # We can cache daily models if needed
        self.xgboost_models = {}
        # Cache market scanner results for the trading day
        self._scanner_cache = None
        self._scanner_cache_date = None

    # ─── MARKET-WIDE RESIDUAL CORRELATION SCANNER ──────────────────────────
    def run_market_scanner(self, top_n: int = TOP_N_CANDIDATES) -> list[str]:
        """
        Scan the full market universe for idiosyncratic momentum using
        Residual Correlation analysis.

        For each stock:
          1. Regress daily returns against SPY (market baseline)
          2. Extract the residual (unexplained by the market)
          3. Sum the last N days of residual = idiosyncratic momentum score
          4. Filter for liquidity and price floor
          5. Rank and return the top N tickers

        Returns: sorted list of top_n ticker symbols with strongest
                 idiosyncratic breakout signals.
        """
        # Cache for the trading day — only scan once per day
        today = datetime.now().date()
        if self._scanner_cache_date == today and self._scanner_cache:
            logger.info(f"[SCANNER] Using cached scan from today ({len(self._scanner_cache)} tickers)")
            return self._scanner_cache

        # 1. Load the universe
        try:
            with open(UNIVERSE_FILE, "r") as f:
                universe = json.load(f)["tickers"]
        except Exception as e:
            logger.error(f"[SCANNER] Failed to load market universe: {e}")
            return []

        # 2. Batch download 30 days of daily data (includes SPY as baseline)
        all_tickers = list(set(universe + ["SPY"]))
        logger.info(f"[SCANNER] Downloading 30d data for {len(all_tickers)} tickers...")

        try:
            raw = yf.download(
                all_tickers,
                period="30d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"[SCANNER] yfinance batch download failed: {e}")
            return []

        if raw.empty:
            logger.error("[SCANNER] Empty batch download — no data returned")
            return []

        # 3. Extract Close prices and Volume
        try:
            closes = raw["Close"]
            volumes = raw["Volume"]
        except KeyError:
            logger.error("[SCANNER] Missing Close/Volume columns in batch data")
            return []

        # Drop tickers with insufficient data
        min_rows = RESIDUAL_LOOKBACK + 5  # Need enough history for regression
        valid_tickers = closes.columns[closes.count() >= min_rows].tolist()
        if "SPY" not in valid_tickers:
            logger.error("[SCANNER] SPY data not available — cannot compute residuals")
            return []

        closes = closes[valid_tickers].ffill()
        volumes = volumes[[t for t in valid_tickers if t in volumes.columns]].ffill()

        # 4. Compute log returns
        log_returns = np.log(closes / closes.shift(1)).dropna()

        if "SPY" not in log_returns.columns:
            logger.error("[SCANNER] SPY returns not computed")
            return []

        spy_returns = log_returns["SPY"]
        stock_tickers = [t for t in log_returns.columns if t != "SPY"]

        # 5. Regress each stock against SPY and compute residual momentum
        scores = {}
        for ticker in stock_tickers:
            try:
                stock_ret = log_returns[ticker].dropna()
                # Align dates between stock and SPY
                common = stock_ret.index.intersection(spy_returns.index)
                if len(common) < min_rows:
                    continue

                y = stock_ret.loc[common].values
                X = sm.add_constant(spy_returns.loc[common].values)
                model = sm.OLS(y, X).fit()

                # Residual = actual return - predicted market return
                residuals = model.resid

                # Sum last N days of residual = idiosyncratic momentum
                residual_momentum = float(residuals[-RESIDUAL_LOOKBACK:].sum())

                # Get latest price and average volume for filtering
                last_price = float(closes[ticker].iloc[-1]) if ticker in closes.columns else 0
                avg_vol = float(volumes[ticker].iloc[-10:].mean()) if ticker in volumes.columns else 0

                # Filter: minimum price and volume
                if last_price < MIN_PRICE:
                    continue
                if avg_vol < MIN_AVG_VOLUME:
                    continue

                scores[ticker] = {
                    "residual_momentum": residual_momentum,
                    "last_price": last_price,
                    "avg_volume": avg_vol,
                    "alpha": float(model.params[0]) * 252,  # Annualized alpha
                    "beta": float(model.params[1]),
                    "r_squared": float(model.rsquared),
                }
            except Exception:
                continue  # Skip problematic tickers silently

        if not scores:
            logger.warning("[SCANNER] No tickers passed filter — returning empty")
            return []

        # 6. Rank by residual momentum (highest = strongest idiosyncratic breakout)
        ranked = sorted(scores.items(), key=lambda x: x[1]["residual_momentum"], reverse=True)
        top = ranked[:top_n]

        # Log the top picks
        logger.info(f"[SCANNER] Top {len(top)} idiosyncratic momentum tickers:")
        for i, (ticker, data) in enumerate(top[:10]):
            logger.info(
                f"  #{i+1} {ticker}: residual_mom={data['residual_momentum']:+.4f}, "
                f"price=${data['last_price']:.2f}, vol={data['avg_volume']:,.0f}, "
                f"α={data['alpha']:+.2f}, β={data['beta']:.2f}"
            )

        result = [t for t, _ in top]

        # Cache for the day
        self._scanner_cache = result
        self._scanner_cache_date = today

        return result

    def analyze(self, tickers: list[str]) -> dict:
        """
        Run the full suite of quantitative models on candidate tickers.
        """
        results = {}
        if not tickers:
            return results

        logger.info(f"Quant Analyst starting evaluation for {len(tickers)} tickers.")

        for ticker in tickers:
            try:
                # 1. GBM Risk Boundaries
                gbm_data = self._run_gbm(ticker)
                
                # 2. Factor Attribution (Alpha vs Beta)
                factor_data = self._run_factor_attribution(ticker)
                
                # 3. XGBoost Directional Probability
                xgb_data = self._run_xgboost(ticker)
                
                results[ticker] = {
                    "gbm": gbm_data,
                    "factors": factor_data,
                    "xgboost": xgb_data,
                }
            except Exception as e:
                logger.warning(f"Quant Analyst failed for {ticker}: {e}")
                results[ticker] = {"error": str(e)}

        return {"quant_data": results}

    def _run_factor_attribution(self, ticker: str) -> dict:
        """Calculate market Beta and idiosyncratic Alpha over the last year."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            raw_data = yf.download([ticker, 'SPY'], start=start_date, end=end_date, auto_adjust=True, progress=False)
            if raw_data.empty or 'Close' not in raw_data.columns:
                 return {"alpha": 0, "beta": 1.0, "r_squared": 0}
                 
            data = raw_data['Close']
            if ticker not in data.columns or 'SPY' not in data.columns:
                return {"alpha": 0, "beta": 1.0, "r_squared": 0}

            returns = data.pct_change().dropna()
            
            y = returns[ticker]
            X = sm.add_constant(returns['SPY'])
            
            model = sm.OLS(y, X).fit()
            
            alpha_annualized = model.params.get('const', 0) * 252
            beta = model.params.get('SPY', 1.0)
            
            return {
                "alpha": float(alpha_annualized),
                "beta": float(beta),
                "r_squared": float(model.rsquared)
            }
        except Exception as e:
            logger.debug(f"Factor attribution error for {ticker}: {e}")
            return {"alpha": 0, "beta": 1.0, "r_squared": 0}

    def _run_gbm(self, ticker: str) -> dict:
        """Monte Carlo GBM to find extreme percentiles (5th/95th) for dynamic stops."""
        try:
            data = yf.download(ticker, period='6mo', auto_adjust=True, progress=False)
            if data.empty:
                return {}
                
            data.columns = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else data.columns
            returns = np.log(1 + data['Close'].pct_change()).dropna()
            
            mu = returns.mean()
            sigma = returns.std()
            last_price = float(data['Close'].iloc[-1])
            
            # Forecast 1 day ahead, 1000 simulations
            num_simulations = 1000
            dt = 1
            drift = (mu - 0.5 * sigma**2) * dt
            
            shocks = np.random.normal(0, 1, num_simulations)
            next_day_prices = last_price * np.exp(drift + sigma * shocks)
            
            p05 = float(np.percentile(next_day_prices, 5))
            p95 = float(np.percentile(next_day_prices, 95))
            
            return {
                "last_price": last_price,
                "volatility": float(sigma),
                "expected_price": float(next_day_prices.mean()),
                "stop_loss_bound_p05": p05,
                "profit_target_bound_p95": p95
            }
        except Exception as e:
            logger.debug(f"GBM error for {ticker}: {e}")
            return {}

    def _run_xgboost(self, ticker: str) -> dict:
        """Calculate probabilistic directional signal for today."""
        try:
            df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 50:
                return {}
            
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns    

            # Features
            window_length = 14
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window_length).mean()
            avg_loss = loss.rolling(window=window_length).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['MA_Dist'] = (df['EMA_20'] - df['EMA_50']) / df['EMA_50']
            
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=14).std()
            df['Vol_Shock'] = df['Volume'].pct_change()

            # Target: 1 day out for day trading context
            forecast_out = 1
            df['Target'] = (df['Close'].shift(-forecast_out) > df['Close']).astype(int)
            df.dropna(inplace=True)

            features = ['RSI', 'MA_Dist', 'Returns', 'Volatility', 'Vol_Shock']
            X = df[features]
            y = df['Target']
            
            # Since this is a fast loop, we quickly fit on all but the last row, 
            # and predict the last row. If performance is an issue, we can cache the models.
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            latest_row = X.tail(1)
            prob_up = float(model.predict_proba(latest_row)[0][1])
            
            signal = "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "HOLD"
            
            return {
                "prob_up": prob_up,
                "prob_down": 1 - prob_up,
                "signal": signal
            }
        except Exception as e:
            logger.debug(f"XGBoost error for {ticker}: {e}")
            return {}

quant_analyst = QuantAnalystAgent()
