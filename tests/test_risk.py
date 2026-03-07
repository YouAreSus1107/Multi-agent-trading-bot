"""
Tests for the Risk Manager Agent — the most critical component.
"""

import pytest
from agents.risk_agent import RiskManagerAgent


class TestRiskManagerAgent:
    """Test all hard-coded kill switches and risk controls."""

    def setup_method(self):
        self.agent = RiskManagerAgent()

    def test_halt_on_zero_equity(self):
        """Should halt when account equity is zero."""
        result = self.agent.evaluate(
            trade_signals=[{"ticker": "XLE"}],
            portfolio={"equity": 0},
            positions=[],
            vix_level=20,
        )
        assert result["halt"] is True

    def test_halt_on_max_daily_loss(self):
        """Should halt when daily loss exceeds 3%."""
        positions = [{"unrealized_pl": -4000}]  # -4% of 100k
        result = self.agent.evaluate(
            trade_signals=[{"ticker": "XLE", "direction": "long", "confidence": 0.8}],
            portfolio={"equity": 100000},
            positions=positions,
            vix_level=20,
        )
        assert result["halt"] is True
        assert "HALT" in result["risk_summary"]

    def test_hedge_mode_on_vix_spike(self):
        """VIX above 35 should trigger hedge mode."""
        result = self.agent.evaluate(
            trade_signals=[{"ticker": "XLE", "direction": "long"}],
            portfolio={"equity": 100000},
            positions=[],
            vix_level=40,
        )
        assert result["hedge_mode"] is True
        assert result["halt"] is False
        assert len(result["hedge_actions"]) > 0

    def test_hedge_actions_include_gold(self):
        """Hedge mode should include buying GLD."""
        result = self.agent.evaluate(
            trade_signals=[],
            portfolio={"equity": 100000},
            positions=[{"ticker": "XLE", "unrealized_pl": 0}],
            vix_level=36,
        )
        gld_actions = [a for a in result["hedge_actions"] if a.get("ticker") == "GLD"]
        assert len(gld_actions) > 0

    def test_reject_low_sharpe(self):
        """Should reject trades with Sharpe < 1.5."""
        result = self.agent.evaluate(
            trade_signals=[{
                "ticker": "XLE",
                "direction": "long",
                "confidence": 0.8,
                "sharpe_ratio": 0.5,
                "expected_return": 0.02,
            }],
            portfolio={"equity": 100000},
            positions=[],
            vix_level=20,
        )
        assert len(result["approved_trades"]) == 0
        assert len(result["rejected_trades"]) > 0
        assert "Sharpe" in result["rejected_trades"][0]["reason"]

    def test_max_position_count(self):
        """Should reject when max 5 positions reached."""
        existing_positions = [
            {"ticker": f"TICK{i}", "unrealized_pl": 0} for i in range(5)
        ]
        result = self.agent.evaluate(
            trade_signals=[{
                "ticker": "NEW",
                "direction": "long",
                "confidence": 0.9,
                "sharpe_ratio": 2.0,
                "expected_return": 0.05,
            }],
            portfolio={"equity": 100000},
            positions=existing_positions,
            vix_level=20,
        )
        assert len(result["approved_trades"]) == 0

    def test_reject_duplicate_ticker(self):
        """Should reject if already holding the same ticker."""
        result = self.agent.evaluate(
            trade_signals=[{
                "ticker": "XLE",
                "direction": "long",
                "confidence": 0.9,
                "sharpe_ratio": 2.0,
                "expected_return": 0.05,
            }],
            portfolio={"equity": 100000},
            positions=[{"ticker": "XLE", "unrealized_pl": 100}],
            vix_level=20,
        )
        rejected = [r for r in result["rejected_trades"] if r["ticker"] == "XLE"]
        assert len(rejected) > 0
        assert "Already holding" in rejected[0]["reason"]

    def test_approved_trade_has_position_sizing(self):
        """Approved trades should include Kelly-based position sizing."""
        result = self.agent.evaluate(
            trade_signals=[{
                "ticker": "XLE",
                "direction": "long",
                "confidence": 0.8,
                "sharpe_ratio": 2.0,
                "expected_return": 0.05,
                "current_price": 90,
            }],
            portfolio={"equity": 100000},
            positions=[],
            vix_level=20,
        )
        assert len(result["approved_trades"]) == 1
        trade = result["approved_trades"][0]
        assert "qty" in trade
        assert "position_dollars" in trade
        assert "kelly_fraction" in trade
        assert trade["qty"] > 0
        assert trade["position_dollars"] > 0
        assert trade["position_pct"] <= 25  # Max 25%

    def test_daily_reset(self):
        """Daily reset should clear all flags."""
        self.agent.halted = True
        self.agent.hedge_mode = True
        self.agent.reset_daily()

        assert self.agent.halted is False
        assert self.agent.hedge_mode is False
        assert self.agent.daily_pnl == 0.0

    def test_normal_operation(self):
        """Normal conditions should pass trades through."""
        result = self.agent.evaluate(
            trade_signals=[{
                "ticker": "GLD",
                "direction": "long",
                "confidence": 0.75,
                "sharpe_ratio": 1.8,
                "expected_return": 0.03,
                "current_price": 180,
            }],
            portfolio={"equity": 50000},
            positions=[],
            vix_level=18,
        )
        assert result["halt"] is False
        assert result["hedge_mode"] is False
        assert len(result["approved_trades"]) == 1
