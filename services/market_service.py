"""
War-Room Bot -- Market Data Service
Uses Finnhub for real-time quotes with Alpaca as fallback for candles.
Also uses Alpaca-compatible VIX proxy for volatility.
"""

import finnhub
import time
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import FINNHUB_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
from utils.logger import get_logger
from utils.rate_limiter import retry_on_rate_limit
from datetime import datetime, timedelta, timezone

logger = get_logger("market_service")


class MarketService:
    """Provides real-time market data via Finnhub + Alpaca fallback."""

    def __init__(self):
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        # Alpaca data client for candles (free, no 403 issues)
        self.alpaca_data = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )

    @retry_on_rate_limit(max_retries=3, initial_backoff=2.0)
    def get_quote(self, ticker: str) -> dict:
        """
        Get a real-time quote for a ticker.
        Tries Finnhub first, falls back to Alpaca.
        """
        # Try Finnhub first
        try:
            q = self.finnhub_client.quote(ticker)
            if q.get("c", 0) and q["c"] > 0:
                return {
                    "current": q.get("c", 0),
                    "high": q.get("h", 0),
                    "low": q.get("l", 0),
                    "open": q.get("o", 0),
                    "prev_close": q.get("pc", 0),
                    "change": q.get("d", 0),
                    "change_pct": q.get("dp", 0),
                    "timestamp": q.get("t", 0),
                }
        except Exception as e:
            logger.warning(f"Finnhub quote failed for {ticker}, trying Alpaca: {e}")

        # Fallback: Alpaca
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = self.alpaca_data.get_stock_latest_quote(request)
            if ticker in quotes:
                q = quotes[ticker]
                price = float(q.ask_price + q.bid_price) / 2 if q.ask_price and q.bid_price else 0
                return {
                    "current": price,
                    "high": price,
                    "low": price,
                    "open": price,
                    "prev_close": price,
                    "change": 0,
                    "change_pct": 0,
                    "timestamp": int(time.time()),
                }
        except Exception as e:
            logger.error(f"Both Finnhub and Alpaca quote failed for {ticker}: {e}")

        return {}

    @retry_on_rate_limit(max_retries=3, initial_backoff=2.0)
    def get_intraday_bars(self, ticker: str, timeframe_minutes: int = 1, days_back: int = 2) -> pd.DataFrame:
        """
        Fetch intraday bars (1-min or 5-min) as a pandas DataFrame.
        Returns DataFrame with columns: [open, high, low, close, volume], datetime-indexed.
        Fetches current + previous session (~780 bars for 1-min, ~156 for 5-min).
        """
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)

        # Primary: Alpaca intraday bars
        try:
            tf = TimeFrame(amount=timeframe_minutes, unit=TimeFrameUnit.Minute)
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start_date,
            )
            bars_response = self.alpaca_data.get_stock_bars(request)

            bar_list = None
            try:
                bar_list = bars_response[ticker]
            except (KeyError, TypeError):
                try:
                    bar_list = list(bars_response.data.get(ticker, []))
                except Exception:
                    pass

            if bar_list and len(bar_list) > 0:
                rows = []
                for b in bar_list:
                    rows.append({
                        "timestamp": b.timestamp,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": int(b.volume),
                    })
                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
                logger.info(f"Alpaca {timeframe_minutes}m bars for {ticker}: {len(df)} rows")
                return df

            logger.warning(f"Alpaca returned empty {timeframe_minutes}m bars for {ticker}")
        except Exception as e:
            logger.warning(f"Alpaca {timeframe_minutes}m bars failed for {ticker}: {e}")

        # Fallback: yfinance
        try:
            import yfinance as yf
            interval = f"{timeframe_minutes}m"
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days_back}d", interval=interval)
            if not hist.empty and len(hist) > 10:
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index.name = "timestamp"
                logger.info(f"yfinance {timeframe_minutes}m bars for {ticker}: {len(df)} rows")
                return df
        except Exception as e:
            logger.warning(f"yfinance {timeframe_minutes}m also failed for {ticker}: {e}")

        logger.error(f"No {timeframe_minutes}m intraday data for {ticker}")
        return empty

    @retry_on_rate_limit(max_retries=3, initial_backoff=2.0)
    def get_candles(
        self, ticker: str, resolution: str = "D",
        from_ts: int = None, to_ts: int = None,
        days_back: int = 60
    ) -> dict:
        """
        Get historical candles. Alpaca primary, Finnhub fallback.
        """
        empty = {"close": [], "high": [], "low": [], "open": [], "volume": [], "timestamp": []}
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)

        # Primary: Alpaca bars
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_date,
            )
            bars_response = self.alpaca_data.get_stock_bars(request)

            # The Alpaca BarSet can be accessed by ticker key or as dict
            bar_list = None
            try:
                bar_list = bars_response[ticker]
            except (KeyError, TypeError):
                # Try iterating the response directly
                try:
                    bar_list = list(bars_response.data.get(ticker, []))
                except Exception:
                    pass

            if bar_list and len(bar_list) > 0:
                result = {
                    "close": [float(b.close) for b in bar_list],
                    "high": [float(b.high) for b in bar_list],
                    "low": [float(b.low) for b in bar_list],
                    "open": [float(b.open) for b in bar_list],
                    "volume": [int(b.volume) for b in bar_list],
                    "timestamp": [int(b.timestamp.timestamp()) for b in bar_list],
                }
                logger.info(f"Alpaca candles for {ticker}: {len(bar_list)} bars")
                return result
            else:
                logger.warning(f"Alpaca returned empty bars for {ticker}")
        except Exception as e:
            logger.warning(f"Alpaca candles failed for {ticker}: {e}")

        # Fallback: Finnhub
        try:
            if to_ts is None:
                to_ts = int(time.time())
            if from_ts is None:
                from_ts = to_ts - (days_back * 86400)

            candles = self.finnhub_client.stock_candles(ticker, resolution, from_ts, to_ts)
            if candles.get("s") == "ok":
                return {
                    "close": candles.get("c", []),
                    "high": candles.get("h", []),
                    "low": candles.get("l", []),
                    "open": candles.get("o", []),
                    "volume": candles.get("v", []),
                    "timestamp": candles.get("t", []),
                }
        except Exception as e:
            logger.warning(f"Finnhub candles also failed for {ticker}: {e}")

        # Fallback 3: yfinance (free, no API key, works for most US stocks)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days_back}d")
            if not hist.empty and len(hist) > 5:
                result = {
                    "close": hist["Close"].tolist(),
                    "high": hist["High"].tolist(),
                    "low": hist["Low"].tolist(),
                    "open": hist["Open"].tolist(),
                    "volume": hist["Volume"].astype(int).tolist(),
                    "timestamp": [int(t.timestamp()) for t in hist.index],
                }
                logger.info(f"yfinance candles for {ticker}: {len(hist)} bars")
                return result
        except Exception as e:
            logger.warning(f"yfinance also failed for {ticker}: {e}")

        logger.error(f"No candle data available for {ticker} (all 3 sources failed)")
        return empty

    @retry_on_rate_limit(max_retries=3, initial_backoff=2.0)
    def get_vix(self) -> float:
        """
        Get the current VIX (Volatility Index) level.
        Uses VIXY ETF price as a proxy since Finnhub free tier
        doesn't support the actual ^VIX index.

        VIXY tracks VIX short-term futures, so we use it as an indicator.
        """
        try:
            # VIXY is the best available VIX proxy on free tiers
            quote = self.get_quote("VIXY")
            vixy_price = quote.get("current", 0)
            if vixy_price > 0:
                # VIXY price ~28 corresponds to VIX ~28
                # This is a rough proxy but directionally accurate
                logger.info(f"VIX proxy (VIXY): {vixy_price}")
                return vixy_price
            return -1.0
        except Exception as e:
            logger.error(f"VIX fetch failed: {e}")
            return -1.0

    def get_multi_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """Get quotes for multiple tickers."""
        quotes = {}
        for ticker in tickers:
            quotes[ticker] = self.get_quote(ticker)
            time.sleep(0.05)  # Small delay
        return quotes

    def detect_risk_regime(self) -> dict:
        """
        Detect if the market is in a 'Risk-Off' regime.
        """
        try:
            spy_quote = self.get_quote("SPY")
            gld_quote = self.get_quote("GLD")
            vix_level = self.get_vix()

            spy_chg = spy_quote.get("change_pct", 0) or 0
            gld_chg = gld_quote.get("change_pct", 0) or 0

            risk_off_signals = 0
            if spy_chg < -0.5:
                risk_off_signals += 1
            if gld_chg > 0.3:
                risk_off_signals += 1
            if vix_level > 20:
                risk_off_signals += 1
            if vix_level > 30:
                risk_off_signals += 1

            if risk_off_signals >= 3:
                regime = "risk-off"
                confidence = 0.9
            elif risk_off_signals >= 2:
                regime = "risk-off"
                confidence = 0.6
            elif spy_chg > 0.5 and gld_chg < 0:
                regime = "risk-on"
                confidence = 0.7
            else:
                regime = "neutral"
                confidence = 0.5

            return {
                "regime": regime,
                "spy_change_pct": spy_chg,
                "gld_change_pct": gld_chg,
                "vix_level": vix_level,
                "confidence": confidence,
            }
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {
                "regime": "neutral",
                "spy_change_pct": 0,
                "gld_change_pct": 0,
                "vix_level": -1,
                "confidence": 0,
            }
