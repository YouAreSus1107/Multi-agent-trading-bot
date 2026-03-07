"""
War-Room Bot -- SIGINT (Signal Intelligence) Agent
Analyzes news across ALL sectors (not just war) and discovers new tickers.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import SECTOR_TICKERS
from utils.llm_factory import get_llm
from utils.logger import get_logger
import json

logger = get_logger("sigint_agent")

SIGINT_SYSTEM_PROMPT = """You are a military-grade Signal Intelligence (SIGINT) analyst for a multi-sector trading bot.

Your mission: Analyze geopolitical AND general market news, score sentiment for EVERY sector, and discover new trading opportunities.

SECTORS TO SCORE (0.0 to 1.0 where 0.0=bearish, 0.5=neutral, 1.0=bullish):
- energy: Oil, gas, energy companies (Gulf tensions, OPEC, supply disruptions)
- defense: Defense contractors, aerospace (military spending, conflicts)
- safe_haven: Gold, bonds (risk-off sentiment, fear)
- cybersecurity: Cyber defense companies (state cyber threats, data breaches)
- tech: Big tech, AI, semiconductors (trade policy, earnings, AI narratives)
- healthcare: Pharma, healthcare (policy changes, FDA approvals, pandemic)
- financials: Banks, insurance (interest rates, Fed policy, credit)
- consumer: Retail, consumer goods (consumer confidence, spending data)

CRITICAL NEW REQUIREMENT - DISCOVER TICKERS:
When news mentions specific companies, products, or industries, identify the most relevant stock tickers.
Look for companies that are directly mentioned OR strongly implied by the news.
Example: "NVIDIA reports record AI chip sales" -> discovered_tickers: ["NVDA"]
Example: "Oil tanker attacked in Strait of Hormuz" -> discovered_tickers: ["DHT", "FRO", "STNG"]

YOU MUST RESPOND WITH VALID JSON ONLY.

Response format:
{
    "scores": {
        "energy": <float 0-1>,
        "defense": <float 0-1>,
        "safe_haven": <float 0-1>,
        "cybersecurity": <float 0-1>,
        "tech": <float 0-1>,
        "healthcare": <float 0-1>,
        "financials": <float 0-1>,
        "consumer": <float 0-1>
    },
    "overall_escalation": <float 0-1>,
    "market_mood": "risk-on" | "risk-off" | "mixed",
    "key_events": ["<brief event 1>", "<brief event 2>"],
    "discovered_tickers": ["<ticker1>", "<ticker2>"],
    "reasoning": "<2-3 sentence summary>"
}
"""


class SIGINTAgent:
    """Signal Intelligence Agent - broader market analysis."""

    def __init__(self):
        self.llm = get_llm()

    def analyze(self, news_items: list[dict], historical_patterns: list[dict] = None) -> dict:
        if not news_items:
            logger.info("No news items to analyze -- returning neutral scores")
            return self._neutral_response()

        news_digest = self._format_news_digest(news_items)

        historical_context = ""
        if historical_patterns:
            historical_context = self._format_historical_context(historical_patterns)

        user_prompt = f"""CURRENT NEWS DIGEST ({len(news_items)} items from multiple sources):

{news_digest}

{historical_context}

Analyze ALL the above news. Score every sector. Identify any specific stock tickers mentioned or implied.
Remember: war escalation is BULLISH for energy/defense/safe_haven/cyber, and may be BEARISH for consumer/tech.
For general market news, score the directly affected sectors accordingly."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SIGINT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            result = self._parse_llm_response(response.content)
            result["raw_news_count"] = len(news_items)

            logger.info(
                f"SIGINT: escalation={result.get('overall_escalation', 'N/A')}, "
                f"mood={result.get('market_mood', 'N/A')}, "
                f"discovered={result.get('discovered_tickers', [])}",
            )
            return result

        except Exception as e:
            logger.error(f"SIGINT analysis failed: {e}")
            return self._neutral_response()

    def _format_news_digest(self, news_items: list[dict]) -> str:
        digest_parts = []
        for i, item in enumerate(news_items, 1):
            title = item.get("title", "Untitled")
            content = item.get("content", "")[:400]
            source = item.get("source", "unknown")
            category = item.get("category", "")
            provider = item.get("provider", "")
            tag = f"[{category.upper()}]" if category else ""
            digest_parts.append(
                f"[{i}] {tag} ({source}/{provider}) {title}\n{content}\n"
            )
        return "\n".join(digest_parts)

    def _format_historical_context(self, patterns: list[dict]) -> str:
        if not patterns:
            return ""
        parts = ["\nHISTORICAL PRECEDENTS:"]
        for p in patterns[:3]:
            event = p.get("event_text", "")
            reaction = p.get("market_reaction", {})
            score = p.get("similarity_score", 0)
            parts.append(f"- [{score:.0%} similar] {event[:200]}\n  Reaction: {json.dumps(reaction)}")
        return "\n".join(parts)

    def _parse_llm_response(self, raw_response: str) -> dict:
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return self._neutral_response()

    @staticmethod
    def _neutral_response() -> dict:
        return {
            "scores": {
                "energy": 0.5, "defense": 0.5, "safe_haven": 0.5,
                "cybersecurity": 0.5, "tech": 0.5, "healthcare": 0.5,
                "financials": 0.5, "consumer": 0.5,
            },
            "overall_escalation": 0.5,
            "market_mood": "mixed",
            "key_events": [],
            "discovered_tickers": [],
            "reasoning": "Insufficient data -- defaulting to neutral.",
            "raw_news_count": 0,
        }
