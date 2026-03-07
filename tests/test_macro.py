"""
Tests for the Macro Analyst Agent.
"""

import pytest


class TestMacroAgent:
    """Test the Macro Analyst Agent."""

    def test_empty_response_structure(self):
        """Empty response should have all required fields."""
        from agents.macro_agent import MacroAgent
        result = MacroAgent._empty_response()

        assert "recommendations" in result
        assert "regime_assessment" in result
        assert "macro_reasoning" in result
        assert result["regime_assessment"] == "neutral"

    def test_no_analysis_without_sentiment(self):
        """Should return empty when no sentiment scores provided."""
        from agents.macro_agent import MacroAgent
        agent = MacroAgent()
        result = agent.analyze({}, {})

        assert result["recommendations"] == []
        assert result["regime_assessment"] == "neutral"

    def test_regime_filter_risk_off_boosts_safe_haven(self):
        """In risk-off regime, safe haven conviction should be boosted."""
        from agents.macro_agent import MacroAgent
        agent = MacroAgent()

        recs = [
            {"ticker": "GLD", "sector": "safe_haven", "conviction": 0.7, "direction": "long"},
            {"ticker": "XLE", "sector": "energy", "conviction": 0.8, "direction": "long"},
        ]

        filtered = agent.apply_regime_filter(recs, "risk-off")

        # GLD should be boosted (0.7 * 1.3 = 0.91)
        gld = next(r for r in filtered if r["ticker"] == "GLD")
        assert gld["conviction"] > 0.7

        # XLE should be dampened (0.8 * 0.7 = 0.56)
        xle_list = [r for r in filtered if r["ticker"] == "XLE"]
        # XLE might be filtered out if dampened below threshold
        if xle_list:
            assert xle_list[0]["conviction"] < 0.8

    def test_regime_filter_risk_on_boosts_energy(self):
        """In risk-on regime, energy/defense conviction should be boosted."""
        from agents.macro_agent import MacroAgent
        agent = MacroAgent()

        recs = [
            {"ticker": "GLD", "sector": "safe_haven", "conviction": 0.7, "direction": "long"},
            {"ticker": "XLE", "sector": "energy", "conviction": 0.7, "direction": "long"},
        ]

        filtered = agent.apply_regime_filter(recs, "risk-on")

        xle = next(r for r in filtered if r["ticker"] == "XLE")
        assert xle["conviction"] > 0.7  # Boosted

    def test_regime_filter_removes_low_conviction(self):
        """Regime filter should remove recommendations below threshold."""
        from agents.macro_agent import MacroAgent
        agent = MacroAgent()

        recs = [
            {"ticker": "GLD", "sector": "safe_haven", "conviction": 0.3, "direction": "long"},
        ]

        filtered = agent.apply_regime_filter(recs, "risk-on")
        # 0.3 * 0.5 = 0.15, below 0.6 threshold
        assert len(filtered) == 0
