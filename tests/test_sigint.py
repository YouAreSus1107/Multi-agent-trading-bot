"""
Tests for the SIGINT Agent.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSIGINTAgent:
    """Test the Signal Intelligence Agent."""

    def test_neutral_response_on_empty_news(self):
        """Agent should return neutral scores when no news is provided."""
        from agents.sigint_agent import SIGINTAgent
        agent = SIGINTAgent()
        result = agent.analyze([])

        assert "scores" in result
        assert result["scores"]["energy"] == 0.5
        assert result["scores"]["defense"] == 0.5
        assert result["scores"]["safe_haven"] == 0.5
        assert result["scores"]["cybersecurity"] == 0.5

    def test_neutral_response_structure(self):
        """Neutral response should have all required fields."""
        from agents.sigint_agent import SIGINTAgent
        result = SIGINTAgent._neutral_response()

        assert "scores" in result
        assert "overall_escalation" in result
        assert "key_events" in result
        assert "reasoning" in result
        assert len(result["scores"]) == 4

    def test_news_digest_formatting(self):
        """News items should be formatted into a readable digest."""
        from agents.sigint_agent import SIGINTAgent
        agent = SIGINTAgent()

        news = [
            {"title": "Iran strikes oil tanker", "content": "Details here", "source": "reuters.com"},
            {"title": "Houthi attacks Red Sea", "content": "Shipping disrupted", "source": "bbc.com"},
        ]

        digest = agent._format_news_digest(news)
        assert "Iran strikes oil tanker" in digest
        assert "Houthi attacks Red Sea" in digest
        assert "[1]" in digest
        assert "[2]" in digest

    def test_parse_valid_json(self):
        """Parser should handle valid JSON response."""
        from agents.sigint_agent import SIGINTAgent
        agent = SIGINTAgent()

        raw = '{"scores": {"energy": 0.9}, "overall_escalation": 0.8, "key_events": [], "reasoning": "test"}'
        result = agent._parse_llm_response(raw)

        assert result["scores"]["energy"] == 0.9
        assert result["overall_escalation"] == 0.8

    def test_parse_markdown_wrapped_json(self):
        """Parser should handle JSON wrapped in markdown code blocks."""
        from agents.sigint_agent import SIGINTAgent
        agent = SIGINTAgent()

        raw = '```json\n{"scores": {"energy": 0.7}, "overall_escalation": 0.6, "key_events": [], "reasoning": "test"}\n```'
        result = agent._parse_llm_response(raw)

        assert result["scores"]["energy"] == 0.7
