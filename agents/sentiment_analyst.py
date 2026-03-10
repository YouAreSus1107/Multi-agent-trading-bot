"""
Sentiment Analyst Agent (Slow Loop — every 5 min)
Scores sentiment from news headlines, tracks momentum.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.logger import get_logger
import json

logger = get_logger("sentiment_analyst")

SYSTEM_PROMPT = """You are the Sentiment Analyst of a day-trading bot.

Your job is to score the SENTIMENT for specific tickers based on news headlines and market context.

For each ticker mentioned or implied in the news:
1. Score sentiment from 0.0 (extremely bearish) to 1.0 (extremely bullish)
2. Track sentiment MOMENTUM: is sentiment getting better or worse?
3. Identify any sentiment EXTREMES (panic selling = contrarian buy, euphoria = top signal)

YOU MUST RESPOND WITH VALID JSON ONLY.

Response format:
{
    "ticker_sentiments": {
        "TICKA": {"score": 0.78, "momentum": "rising", "note": "Defense contract buzz"},
        "TICKB": {"score": 0.55, "momentum": "flat", "note": "Mixed AI regulation news"}
    },
    "market_mood": "cautious",
    "fear_greed_estimate": 35,
    "contrarian_signals": ["TICKC oversold on overreaction"],
    "summary": "Overall market cautious but defense names seeing bullish sentiment momentum"
}

market_mood values: "euphoria", "bullish", "cautious", "fearful", "panic"
momentum values: "surging", "rising", "flat", "falling", "crashing"
"""


class SentimentAnalyst:
    """
    Slow-loop analyst: scores sentiment per ticker from news.
    Runs every 5 minutes.
    """

    def __init__(self):
        self.llm = get_llm("parsing")
        self._prev_sentiments = {}  # Track for momentum

    def analyze(self, news_events: list, target_tickers: list[str]) -> dict:
        """
        Score sentiment for tickers based on news events.
        """
        if not news_events and not target_tickers:
            return self._empty_result()

        # Build news summary
        event_text = ""
        if news_events:
            event_text = "\n".join([
                f"- {e.get('headline', e.get('title', 'Unknown'))}: "
                f"impact={e.get('impact', 'unknown')}, "
                f"tickers={e.get('affected_tickers', [])}, "
                f"direction={e.get('direction', 'unknown')}"
                for e in news_events[:15]
            ])

        prev_text = ""
        if self._prev_sentiments:
            prev_text = f"\nPREVIOUS SENTIMENT SCORES (for momentum tracking):\n{json.dumps(self._prev_sentiments, indent=2)}"

        prompt = f"""Score sentiment for these day-trading tickers based on the news.

TARGET TICKERS: {', '.join(target_tickers[:20])}

NEWS EVENTS:
{event_text or 'No specific events — use general market conditions.'}
{prev_text}

Focus on tickers that show STRONG directional sentiment (above 0.7 or below 0.3).
Track momentum vs previous scores."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = self._parse(response.content)

            # Store for next cycle momentum tracking
            self._prev_sentiments = result.get("ticker_sentiments", {})

            logger.info(
                f"Sentiment Analyst: {len(result.get('ticker_sentiments', {}))} tickers scored, "
                f"mood={result.get('market_mood', 'unknown')}"
            )
            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "ticker_sentiments": {},
            "market_mood": "unknown",
            "fear_greed_estimate": 50,
            "contrarian_signals": [],
            "summary": "No data",
        }

    def _parse(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return self._empty_result()
