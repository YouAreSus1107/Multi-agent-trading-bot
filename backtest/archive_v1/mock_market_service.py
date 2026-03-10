import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from utils.quant_engine import precompute_mtf_tsl_for_ticker, precompute_quant_metrics_for_ticker

class MockMarketService:
    """
    A drop-in replacement for MarketService that reads from local CSVs
    instead of hitting the live Alpaca/Finnhub APIs.
    """
    def __init__(self, current_time: datetime, data_file: str = None, daily_file: str = None):
        self.current_time = current_time
        
        # Load datasets into memory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        hist_path = data_file if data_file else os.path.join(data_dir, "historical_bars.csv")
        augmented_path = os.path.join(os.path.dirname(hist_path), "augmented_" + os.path.basename(hist_path))
        
        if os.path.exists(augmented_path):
            df_intraday = pd.read_csv(augmented_path)
            df_intraday["timestamp"] = pd.to_datetime(df_intraday["timestamp"], utc=True)
            df_intraday.set_index("timestamp", inplace=True)
            df_intraday.sort_index(inplace=True)
            precomputed_groups = {ticker: group for ticker, group in df_intraday.groupby("symbol")}
            self._intraday_by_ticker = precomputed_groups
        else:
            df_intraday = pd.read_csv(hist_path)
            df_intraday["timestamp"] = pd.to_datetime(df_intraday["timestamp"], utc=True)
            df_intraday.set_index("timestamp", inplace=True)
            df_intraday.sort_index(inplace=True)
            
            # Precompute the MTF positions and Quant Metrics heavily for the entire dataset upfront so it doesnt re-loop per 5 minutes
            print("Precomputing Pine Script MTF & Quant Indicators for dataset (should take a few seconds)...")
            precomputed_groups = {}
            augmented_dfs = []
            for ticker, group in df_intraday.groupby("symbol"):
                group = precompute_mtf_tsl_for_ticker(group)
                group = precompute_quant_metrics_for_ticker(group)
                precomputed_groups[ticker] = group
                augmented_dfs.append(group)
                
            self._intraday_by_ticker = precomputed_groups
            
            # Save the augmented dataset to avoid recomputation next time
            try:
                full_augmented_df = pd.concat(augmented_dfs)
                full_augmented_df.to_csv(augmented_path, index=True)
                print(f"Saved precomputed metrics to {os.path.basename(augmented_path)} for instant loads.")
            except Exception as e:
                print(f"Failed to cache augmented dataset: {e}")
        
        daily_path = daily_file if daily_file else os.path.join(data_dir, "daily_bars.csv")
        df_daily = pd.read_csv(daily_path)
        df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"], utc=True)
        df_daily.set_index("timestamp", inplace=True)
        df_daily.sort_index(inplace=True)
        self._daily_by_ticker = {t: g for t, g in df_daily.groupby("symbol")}
        
    def set_time(self, new_time: datetime):
        """Advances the internal clock of the mock service."""
        self.current_time = new_time

    def get_quote(self, ticker: str) -> dict:
        """
        Returns the last known price for the ticker at the current virtual time.
        """
        if ticker not in self._intraday_by_ticker:
            return {}
            
        group = self._intraday_by_ticker[ticker]
        df = group.loc[:self.current_time]
        
        if df.empty:
            return {}
            
        latest = df.iloc[-1]
        price = float(latest["close"])
        timestamp = df.index[-1].timestamp()
        
        return {
            "current": price,
            "high": price,
            "low": price,
            "open": price,
            "prev_close": price,
            "change": 0,
            "change_pct": 0,
            "timestamp": int(timestamp),
        }

    def get_latest_bar(self, ticker: str):
        """
        Returns the latest precomputed bar at or before current_time as a pandas Series.
        Uses binary search for O(log n) lookup — no DataFrame slicing or copying.
        Returns None if no data available.
        """
        if ticker not in self._intraday_by_ticker:
            return None
        group = self._intraday_by_ticker[ticker]
        idx = group.index.searchsorted(self.current_time, side='right')
        if idx == 0:
            return None
        return group.iloc[idx - 1]

    def get_intraday_bars(self, ticker: str, timeframe_minutes: int = 5, days_back: int = 2) -> pd.DataFrame:
        """
        Returns intraday bars cut off EXACTLY at `self.current_time` to prevent look-ahead bias.
        """
        if ticker not in self._intraday_by_ticker:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "mtf_total_pos", "tsl_1"])
            
        start_date = self.current_time - timedelta(days=days_back)
        group = self._intraday_by_ticker[ticker]
        df = group.loc[start_date:self.current_time]
        
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            
        # Keep augmented columns (dynamic quant caches included if present)
        cols_to_keep = ["open", "high", "low", "close", "volume", "mtf_total_pos", "tsl_1", 
                        "atr_5m", "vwap", "vwap_zscore", "sigma", "buy_pressure", 
                        "sell_pressure", "delta_ratio", "smart_bounce", "last_bar_buy_pct"]
        cols_present = [c for c in cols_to_keep if c in df.columns]
        return df[cols_present].copy()

    def get_candles(self, ticker: str, resolution: str = "D", from_ts: int = None, to_ts: int = None, days_back: int = 60) -> dict:
        """
        Returns daily candles cut off at `self.current_time`.
        """
        empty = {"close": [], "high": [], "low": [], "open": [], "volume": [], "timestamp": []}
        
        if ticker not in self._daily_by_ticker:
            return empty
            
        start_date = self.current_time - timedelta(days=days_back)
        group = self._daily_by_ticker[ticker]
        df = group.loc[start_date:self.current_time]
        
        if df.empty:
            return empty
            
        return {
            "close": df["close"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "open": df["open"].tolist(),
            "volume": df["volume"].astype(int).tolist(),
            "timestamp": [int(t.timestamp()) for t in df.index],
        }

    def get_vix(self) -> float:
        return 20.0  # Dummy value to avoid API lookups for backtest

    def detect_risk_regime(self) -> dict:
        return {
            "regime": "neutral",
            "spy_change_pct": 0,
            "gld_change_pct": 0,
            "vix_level": 20.0,
            "confidence": 0.5,
        }
