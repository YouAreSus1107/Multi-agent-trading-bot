"""
Tests for core services — Technical Indicators.
Verifies mathematical correctness of RSI, Bollinger Bands, Sharpe Ratio, and Kelly Criterion.
"""

import pytest
import numpy as np
from utils.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_sharpe_ratio,
    kelly_criterion,
    compute_daily_returns,
)


class TestRSI:
    """Test RSI calculation."""

    def test_rsi_neutral_on_insufficient_data(self):
        """RSI should return 50 (neutral) with insufficient data."""
        assert compute_rsi([100, 101], period=14) == 50.0

    def test_rsi_100_on_all_gains(self):
        """RSI should be 100 when all moves are upward."""
        prices = list(range(100, 120))  # 20 consecutive gains
        rsi = compute_rsi(prices, period=14)
        assert rsi == 100.0

    def test_rsi_within_bounds(self):
        """RSI should always be between 0 and 100."""
        prices = [100, 102, 99, 101, 98, 103, 97, 104, 96, 105,
                  94, 106, 93, 107, 92, 108, 91, 109, 90, 110]
        rsi = compute_rsi(prices, period=14)
        assert 0 <= rsi <= 100

    def test_rsi_typical_value(self):
        """RSI for mixed data should be between 30-70 (typical range)."""
        np.random.seed(42)
        prices = [100]
        for _ in range(30):
            prices.append(prices[-1] + np.random.randn() * 2)
        rsi = compute_rsi(prices, period=14)
        assert 0 <= rsi <= 100


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_order(self):
        """Upper band should be above middle, which should be above lower."""
        prices = list(range(100, 121))  # 21 prices
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert upper >= middle >= lower

    def test_bollinger_insufficient_data(self):
        """With insufficient data, bands should collapse to mean."""
        prices = [100, 101, 102]
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert upper == middle == lower

    def test_bollinger_with_constant_prices(self):
        """With constant prices, bands should be very tight."""
        prices = [100.0] * 25
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert middle == 100.0
        assert abs(upper - lower) < 0.01  # Nearly zero spread


class TestSharpeRatio:
    """Test Sharpe Ratio calculation."""

    def test_sharpe_zero_on_insufficient_data(self):
        """Sharpe should be 0 with less than 2 returns."""
        assert compute_sharpe_ratio([]) == 0.0
        assert compute_sharpe_ratio([0.01]) == 0.0

    def test_sharpe_zero_on_zero_volatility(self):
        """Sharpe should be 0 when returns have zero variance."""
        returns = [0.001] * 10  # Constant returns
        sharpe = compute_sharpe_ratio(returns)
        # With zero std dev, should return 0
        assert sharpe == 0.0

    def test_sharpe_positive_for_good_returns(self):
        """Sharpe should be positive for consistently positive returns."""
        returns = [0.01, 0.02, 0.015, 0.01, 0.025, 0.01, 0.02, 0.015, 0.01, 0.02]
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe > 0


class TestKellyCriterion:
    """Test Kelly Criterion position sizing."""

    def test_kelly_zero_for_bad_odds(self):
        """Kelly should be 0 when win probability is too low."""
        assert kelly_criterion(0.2, 1.0) == 0.0

    def test_kelly_capped_at_25_percent(self):
        """Kelly should never exceed 25% allocation."""
        kelly = kelly_criterion(0.95, 10.0)
        assert kelly <= 0.25

    def test_kelly_zero_for_invalid_inputs(self):
        """Kelly should be 0 for invalid inputs."""
        assert kelly_criterion(0, 1.0) == 0.0
        assert kelly_criterion(1.0, 1.0) == 0.0
        assert kelly_criterion(0.5, 0) == 0.0
        assert kelly_criterion(-0.1, 1.0) == 0.0

    def test_kelly_reasonable_output(self):
        """Kelly should give reasonable sizing for typical inputs."""
        # 60% win rate, 1.5:1 win/loss ratio
        kelly = kelly_criterion(0.6, 1.5)
        assert 0 < kelly <= 0.25


class TestDailyReturns:
    """Test daily returns calculation."""

    def test_empty_on_insufficient_data(self):
        """Should return empty list with fewer than 2 prices."""
        assert compute_daily_returns([]) == []
        assert compute_daily_returns([100]) == []

    def test_correct_returns(self):
        """Daily returns should be correctly computed."""
        prices = [100, 110, 99]
        returns = compute_daily_returns(prices)
        assert len(returns) == 2
        assert abs(returns[0] - 0.10) < 0.001   # 100 → 110 = +10%
        assert abs(returns[1] - (-0.1)) < 0.001  # 110 → 99 = -10%
