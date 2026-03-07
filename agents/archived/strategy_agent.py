"""
War-Room Bot -- Strategy & Reasoning Agent (The "Brain")
Deep technical analysis + news + memory. Generates BOTH entry AND exit signals.
Requires multiple bearish confirmations before exiting. Enforces minimum hold period.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import STRATEGY_CONFIG, RISK_CONFIG
from utils.indicators import (
    compute_trend_score, compute_daily_returns,
    compute_sharpe_ratio,
)
from utils.llm_factory import get_llm
from utils.memory import get_memory
from utils.logger import get_logger
import json

logger = get_logger("strategy_agent")

# Minimum cycles to hold before considering exit (prevents buy-then-sell whipsaw)
MIN_HOLD_CYCLES = 3

STRATEGY_SYSTEM_PROMPT = """You are the Strategy Agent of a trading bot. You combine NEWS SENTIMENT + DEEP TECHNICAL ANALYSIS + MEMORY OF PAST DECISIONS.

## ENTRY RULES (be willing to buy if conditions are met):
You need BOTH sentiment AND technical support to enter:
1. Sector sentiment > 0.45
2. Trend score > 50 (composite indicator from MACD, RSI, EMA, Bollinger, volume)
3. If trend score > 60, you can buy even if sentiment is borderline (> 0.35)
4. Volume spikes with positive momentum = strong buy signal

IMPORTANT: Be WILLING to enter trades. The bot should actively find opportunities. If technicals are neutral (score 45-55) but sentiment is strong (> 0.6), lean toward buying.

## EXIT RULES (be PATIENT — don't sell too quickly):
You need AT LEAST 2 out of these 5 bearish signals to recommend selling:
1. RSI > 80 (strongly overbought — NOT just 70)
2. MACD bearish crossover (clear trend reversal)
3. EMA death cross (9-day crosses below 21-day)  
4. Trend score < 35 (multiple indicators are bearish)
5. Position has lost > 2.5% (stop loss)

CRITICAL: Do NOT sell a position just because ONE indicator looks slightly negative.
- RSI at 72 is NOT enough to sell by itself
- A single MACD shift is NOT enough to sell by itself
- If the original thesis for buying still holds, KEEP the position

If a position is up > 5%, consider taking partial profit but don't necessarily close it all.

## MEMORY CONTEXT
You will receive your past reasoning and trade history. Use it to:
- Stay consistent with your thesis unless conditions CLEARLY changed
- Avoid flip-flopping (buying then selling the same ticker repeatedly)
- Learn from past trades

YOU MUST RESPOND WITH VALID JSON ONLY.

Response format:
{
    "trade_signals": [
        {
            "ticker": "XLE",
            "direction": "long",
            "confidence": 0.82,
            "expected_return": 0.035,
            "technical_confirmation": true,
            "reasoning": "Trend score 68, MACD bullish crossover, RSI 45, energy sentiment 0.85"
        }
    ],
    "exit_signals": [
        {
            "ticker": "GLD",
            "action": "sell",
            "reason": "RSI at 82 + MACD bearish crossover (2/5 bearish signals)",
            "urgency": "high"
        }
    ],
    "rejected_trades": [
        {
            "ticker": "LMT",
            "reason": "Trend score 42 -- below threshold"
        }
    ],
    "hold_confirmations": [
        {
            "ticker": "XLE",
            "reason": "Thesis still valid -- energy sentiment strong, RSI 55"
        }
    ],
    "overall_conviction": 0.75,
    "cycle_summary": "Entered XLE on energy sentiment, holding GLD thesis intact"
}
"""


class StrategyAgent:
    """Strategy Agent with deep technical analysis, exit logic, and cycle memory."""

    def __init__(self):
        self.llm = get_llm()
        self.memory = get_memory()

    def analyze(
        self,
        recommendations: list[dict],
        market_data: dict,
        sentiment_scores: dict,
        held_positions: list[dict] = None,
        current_cycle: int = 0,
    ) -> dict:
        """
        Analyze recommendations AND held positions.
        Uses memory for context and enforces minimum hold period.
        """
        if not recommendations and not held_positions:
            return {"trade_signals": [], "exit_signals": [], "hold_confirmations": [],
                    "rejected_trades": [], "overall_conviction": 0, "cycle_summary": "No activity"}

        # Compute technical analysis for each ticker
        tech_profiles = {}

        # Analyze recommended tickers for entry
        for rec in (recommendations or []):
            ticker = rec.get("ticker", "")
            if ticker and ticker in market_data:
                tech_profiles[ticker] = self._compute_profile(market_data[ticker])

        # Analyze held positions for exits
        for pos in (held_positions or []):
            ticker = pos.get("ticker", "")
            if ticker and ticker in market_data and ticker not in tech_profiles:
                tech_profiles[ticker] = self._compute_profile(market_data[ticker])

        # ENFORCE MINIMUM HOLD: filter out positions too young to sell
        eligible_for_exit = []
        too_young_to_sell = []
        for pos in (held_positions or []):
            ticker = pos.get("ticker", "")
            cycles_held = self.memory.get_cycles_held(ticker, current_cycle)
            if cycles_held >= MIN_HOLD_CYCLES:
                eligible_for_exit.append(pos)
            else:
                too_young_to_sell.append(ticker)
                logger.info(
                    f"[HOLD] {ticker} too young to sell (held {cycles_held}/{MIN_HOLD_CYCLES} cycles)"
                )

        # Get memory context for LLM
        memory_context = self.memory.get_context_for_llm(current_cycle)

        # Build context for LLM
        tech_summary = json.dumps({
            ticker: {
                "trend_score": p.get("score", 50),
                "recommendation": p.get("recommendation", "hold"),
                "rsi": p.get("rsi", 50),
                "macd_trend": p.get("macd", {}).get("trend", "neutral"),
                "ema_signal": p.get("ema_cross", {}).get("signal", "neutral"),
                "bollinger_position": p.get("bollinger_position", "middle"),
                "volume_spike": p.get("volume_spike", False),
                "sharpe": p.get("sharpe", 0),
            }
            for ticker, p in tech_profiles.items()
        }, indent=2)

        held_summary = json.dumps([
            {
                "ticker": p.get("ticker", ""),
                "qty": p.get("qty", 0),
                "entry_price": p.get("entry_price", 0),
                "current_price": p.get("current_price", 0),
                "unrealized_pl_pct": p.get("unrealized_pl_pct", 0),
                "cycles_held": self.memory.get_cycles_held(p.get("ticker", ""), current_cycle),
                "original_thesis": self.memory.get_position_thesis(p.get("ticker", "")).get("thesis", "unknown"),
            }
            for p in eligible_for_exit
        ], indent=2)

        user_prompt = f"""MEMORY (your past reasoning):
{memory_context}

SECTOR SENTIMENT (from SIGINT):
{json.dumps(sentiment_scores, indent=2)}

TICKER RECOMMENDATIONS (from Macro):
{json.dumps([{"ticker": r.get("ticker"), "sector": r.get("sector"), "direction": r.get("direction"), "conviction": r.get("conviction")} for r in (recommendations or [])], indent=2)}

DEEP TECHNICAL ANALYSIS:
{tech_summary}

POSITIONS ELIGIBLE FOR EXIT (held >= {MIN_HOLD_CYCLES} cycles):
{held_summary}

POSITIONS TOO YOUNG TO SELL: {too_young_to_sell}

Rules:
1. Generate ENTRY signals where sentiment + technicals align. Be willing to buy.
2. For exits: need 2+ out of 5 bearish signals. Don't sell on a single weak indicator.
3. For held positions, confirm the thesis is still valid or recommend exit.
4. Write a brief cycle_summary of your decisions."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            result = self._parse_response(response.content)

            # Attach technical data to signals
            for signal in result.get("trade_signals", []):
                ticker = signal.get("ticker", "")
                if ticker in tech_profiles:
                    signal["trend_score"] = tech_profiles[ticker].get("score", 50)
                    signal["sharpe_ratio"] = tech_profiles[ticker].get("sharpe", 0)

            # Record cycle summary in memory
            cycle_summary = result.get("cycle_summary", "")
            if cycle_summary:
                self.memory.record_cycle_summary(current_cycle, cycle_summary)

            logger.info(
                f"Strategy: {len(result.get('trade_signals', []))} entries, "
                f"{len(result.get('exit_signals', []))} exits, "
                f"{len(result.get('hold_confirmations', []))} holds, "
                f"{len(result.get('rejected_trades', []))} rejected"
            )
            return result

        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {"trade_signals": [], "exit_signals": [], "hold_confirmations": [],
                    "rejected_trades": [], "overall_conviction": 0, "cycle_summary": "Error"}

    def _compute_profile(self, candles: dict) -> dict:
        """Compute full technical profile for a ticker."""
        closes = candles.get("close", [])
        volumes = candles.get("volume", [])
        highs = candles.get("high", [])
        lows = candles.get("low", [])

        if closes and len(closes) >= 5:
            profile = compute_trend_score(closes, volumes, highs, lows)
            returns = compute_daily_returns(closes)
            profile["sharpe"] = compute_sharpe_ratio(returns)
            return profile

        return {"score": 50, "recommendation": "hold", "rsi": 50,
                "macd": {"trend": "neutral"}, "ema_cross": {"signal": "neutral"},
                "bollinger_position": "middle", "volume_spike": False, "sharpe": 0}

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
            return {"trade_signals": [], "exit_signals": [], "hold_confirmations": [],
                    "rejected_trades": [], "overall_conviction": 0, "cycle_summary": "Parse error"}
