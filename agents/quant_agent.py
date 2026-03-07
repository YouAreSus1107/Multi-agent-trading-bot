"""
Quant Analyst Agent
Mathematical engine for Factor Attribution, XGBoost Probability, and GBM Risk Modeling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import norm
import xgboost as xgb
from datetime import datetime, timedelta
import warnings

from utils.logger import get_logger

logger = get_logger("quant_analyst")

# Suppress warnings from yfinance and statsmodels
warnings.filterwarnings('ignore')

class QuantAnalystAgent:
    """Pre-market quantitative mathematical modeling."""

    def __init__(self):
        # We can cache daily models if needed
        self.xgboost_models = {}

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
