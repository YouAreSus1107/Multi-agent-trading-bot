"""
War-Room Bot -- Macro Analyst Agent
Maps sentiment to tickers AND incorporates discovered tickers from SIGINT.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import SECTOR_TICKERS, STRATEGY_CONFIG
from utils.llm_factory import get_llm
from utils.logger import get_logger
import json

logger = get_logger("macro_agent")

MACRO_SYSTEM_PROMPT = """You are a Macro Analyst for a multi-sector trading bot. Map sentiment scores to specific tradeable tickers.

AVAILABLE SECTORS AND TICKERS:
- energy: XLE, USO, CVX, XOM, OXY, COP, SLB
- defense: LMT, RTX, NOC, GD, BA, HII, LHX
- safe_haven: GLD, TLT, SLV, UUP, BND
- cybersecurity: PANW, CRWD, ZS, FTNT, NET
- tech: QQQ, MSFT, AAPL, NVDA, GOOGL, META, AMZN, AMD
- healthcare: XLV, UNH, JNJ, PFE, ABT, LLY, MRK
- financials: XLF, JPM, BAC, GS, MS, V
- consumer: XLY, AMZN, TSLA, NKE, SBUX, HD, WMT

RULES:
1. Recommend tickers from sectors with sentiment >= 0.45 (bullish) or <= 0.35 (bearish for shorts)
2. ALSO include any "discovered_tickers" from SIGINT if they are valid US stocks
3. For bearish sectors (score < 0.35), recommend SHORT positions
4. Diversify: recommend from MULTIPLE sectors, not just the strongest
5. Prioritize ETFs (XLE, QQQ, XLV) for broader exposure, individual stocks for specific catalysts
6. Maximum 8 recommendations per cycle

YOU MUST RESPOND WITH VALID JSON ONLY.

Response format:
{
    "recommendations": [
        {
            "ticker": "XLE",
            "sector": "energy",
            "direction": "long",
            "conviction": 0.85,
            "reasoning": "Gulf escalation bullish for energy"
        }
    ],
    "regime_assessment": "risk-off" | "risk-on" | "neutral",
    "macro_reasoning": "<2-3 sentence macro view>"
}
"""


class MacroAgent:
    """Macro Analyst -- maps sentiment to tickers across all sectors."""

    def __init__(self):
        self.llm = get_llm()

    def analyze(
        self,
        sentiment_scores: dict,
        regime_data: dict,
        key_events: list[str] = None,
        discovered_tickers: list[str] = None,
    ) -> dict:
        if not sentiment_scores:
            return self._empty_response()

        user_prompt = f"""SECTOR SENTIMENT SCORES (from SIGINT):
{json.dumps(sentiment_scores, indent=2)}

MARKET REGIME:
- Regime: {regime_data.get('regime', 'unknown')}
- SPY change: {regime_data.get('spy_change_pct', 'N/A')}%
- GLD change: {regime_data.get('gld_change_pct', 'N/A')}%
- VIX level: {regime_data.get('vix_level', 'N/A')}

KEY EVENTS: {json.dumps(key_events or [])}

DISCOVERED TICKERS (from news analysis): {json.dumps(discovered_tickers or [])}

Recommend specific tickers to trade. Include discovered tickers if they align with sentiment.
Diversify across multiple sectors. Use ETFs for broad exposure."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=MACRO_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            result = self._parse_response(response.content)
            logger.info(
                f"Macro: {len(result.get('recommendations', []))} recommendations, "
                f"regime={result.get('regime_assessment', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return self._empty_response()

    def apply_regime_filter(self, recommendations: list[dict], regime: str) -> list[dict]:
        filtered = []
        for rec in recommendations:
            sector = rec.get("sector", "")
            conviction = rec.get("conviction", 0)

            if regime == "risk-off":
                if sector == "safe_haven":
                    rec["conviction"] = min(1.0, conviction * 1.3)
                elif sector in ("energy", "defense"):
                    rec["conviction"] = conviction * 0.8
                elif sector in ("consumer", "tech"):
                    rec["conviction"] = conviction * 0.6
            elif regime == "risk-on":
                if sector == "safe_haven":
                    rec["conviction"] = conviction * 0.5
                elif sector in ("tech", "consumer"):
                    rec["conviction"] = min(1.0, conviction * 1.2)

            if rec["conviction"] >= STRATEGY_CONFIG["min_sentiment_score"]:
                filtered.append(rec)

        return sorted(filtered, key=lambda x: x["conviction"], reverse=True)

    def _parse_response(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return self._empty_response()

    @staticmethod
    def _empty_response() -> dict:
        return {
            "recommendations": [],
            "regime_assessment": "neutral",
            "macro_reasoning": "Insufficient data.",
        }
