"""
UniverseService — Daily S&P 500 Universe Ranking for Paper Trading
Wraps the backtest's rank_universe() for live use, cached per day.

Layer 1 of the 3-layer backtest architecture:
  - Ranks the full S&P 500 universe by alpha/momentum/volatility (T-1 data).
  - Called once per day at startup / first cycle.
  - Outputs top_n tickers for the TechnicalAnalyst to analyze.

Note: Regime switching is not yet implemented.
      Currently always assumes 'bull' (momentum model only).
"""

import os
import pandas as pd
from datetime import datetime, date, timedelta
from utils.logger import get_logger

logger = get_logger("universe_service")


class UniverseService:
    """
    Provides the daily-ranked S&P 500 ticker list using the same
    `rank_universe()` logic from the backtest engine (T-1 data, no look-ahead).
    """

    def __init__(self):
        self._cached_tickers: list[str] = []
        self._cached_date: date | None = None
        self._all_tickers: dict = {}  # Loaded once, reused all day

    def _load_daily_data(self) -> dict:
        """Load S&P 500 daily data (expensive — done once per session)."""
        if self._all_tickers:
            return self._all_tickers

        try:
            from backtest.data_loader_v2 import parse_sp500_daily
            logger.info("[UNIVERSE] Loading S&P 500 daily data (one-time)...")
            self._all_tickers = parse_sp500_daily(fetch_yfinance=False)
            logger.info(f"[UNIVERSE] Loaded {len(self._all_tickers)} tickers")
        except Exception as e:
            logger.error(f"[UNIVERSE] Failed to load S&P 500 daily data: {e}")
            self._all_tickers = {}

        return self._all_tickers

    def get_todays_universe(self, top_n: int = 15) -> list[str]:
        """
        Return the daily-ranked universe for today.
        Ranked using ONLY T-1 data to prevent look-ahead bias.
        Cached for the full trading day.
        """
        today = date.today()

        # Return cache if already computed today
        if self._cached_date == today and self._cached_tickers:
            logger.info(f"[UNIVERSE] Using cached universe ({len(self._cached_tickers)} tickers): {self._cached_tickers[:5]}...")
            return self._cached_tickers

        try:
            all_tickers = self._load_daily_data()
            if not all_tickers:
                logger.warning("[UNIVERSE] No daily data loaded — returning empty universe")
                return []

            from backtest.data_loader_v2 import rank_universe

            # Use T-1: yesterday's close as the cutoff to avoid look-ahead
            as_of_date = pd.Timestamp(today) - pd.Timedelta(days=1)
            # Walk back to last weekday if needed
            while as_of_date.weekday() >= 5:
                as_of_date -= pd.Timedelta(days=1)

            logger.info(f"[UNIVERSE] Ranking universe as of {as_of_date.date()} (T-1)...")

            top_picks = rank_universe(
                all_tickers,
                as_of_date=as_of_date,
                lookback_days=252,
                ranking_window=30,
                top_n=top_n,
            )

            self._cached_tickers = [p["ticker"] for p in top_picks]
            self._cached_date = today

            logger.info(
                f"[UNIVERSE] Today's top {len(self._cached_tickers)} ranked tickers: "
                f"{self._cached_tickers}"
            )

        except Exception as e:
            logger.error(f"[UNIVERSE] rank_universe() failed: {e}")
            self._cached_tickers = []

        return self._cached_tickers

    def get_todays_regime(self) -> str:
        """
        Returns the current market regime.
        Regime switching not yet implemented — always returns 'bull'
        (momentum-only model).
        """
        return "bull"

    def invalidate_cache(self):
        """Force a fresh ranking on the next call (e.g. after new day detection)."""
        self._cached_date = None
        self._cached_tickers = []


# Module-level singleton
universe_service = UniverseService()
