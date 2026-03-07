"""
Tests for the Strategy & Reasoning Agent.
"""

import pytest


class TestStrategyAgent:
    """Test the Strategy Agent."""

    def test_no_signals_without_recommendations(self):
        """Should produce no signals when there are no ticker recommendations."""
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()
        result = agent.analyze([], {}, {})

        assert result["trade_signals"] == []
        assert result["rejected_trades"] == []
        assert result["overall_conviction"] == 0

    def test_prompt_building(self):
        """Prompt should include all necessary data."""
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()

        recs = [{"ticker": "XLE", "direction": "long", "conviction": 0.8}]
        technicals = {"XLE": {"rsi": 55, "sharpe_ratio": 1.8}}
        sentiment = {"energy": 0.85}

        prompt = agent._build_prompt(recs, technicals, sentiment)

        assert "XLE" in prompt
        assert "energy" in prompt
        assert "0.85" in prompt

    def test_parse_valid_response(self):
        """Should parse valid JSON response correctly."""
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()

        raw = '{"trade_signals": [{"ticker": "XLE", "direction": "long"}], "rejected_trades": [], "overall_conviction": 0.8}'
        result = agent._parse_response(raw)

        assert len(result["trade_signals"]) == 1
        assert result["trade_signals"][0]["ticker"] == "XLE"

    def test_parse_invalid_response(self):
        """Should return empty signals on invalid JSON."""
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()

        result = agent._parse_response("this is not json")
        assert result["trade_signals"] == []
