"""
Research Team Agent — Bull/Bear Structured Debate
Critically assesses analyst insights through structured debate.
Produces trade decisions with high-confidence reasoning.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.memory import get_memory
from utils.logger import get_logger
import json
import os
from datetime import datetime, timezone
import concurrent.futures

logger = get_logger("research_team")

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "research_log.jsonl")

BULL_PROMPT = """You are the BULL RESEARCHER on a day-trading team.
Your job is to build the STRONGEST CASE FOR buying each ticker.

PRO STRATEGIES TO LOOK FOR (these are HIGH-CONVICTION setups):
1. RIDE THE TREND: If EMA100_BULLISH signal present, ONLY go long. High-timeframe bias is king.
2. VWAP_SMART_BOUNCE: Price near VWAP with long wick + high volume = smart money stepping in. Strong buy.
3. STOP_HUNT_LONG: Price wicked below previous low but recovered = retail stops got hunted. Reversal long.
4. MM_REFILL_ACCUMULATION: Volume spike with no price move = big player filling. Follow the smart money.
5. RSI_EXHAUSTION_LONG: RSI sub-10 with reversal candle = rubber band ready to snap. Contrarian long.
6. OVERSOLD + trend aligned: Don't just buy oversold, make sure the trend supports it.
7. News catalyst + technical confirmation = highest conviction trades.

For each candidate, argue why the price will go UP today. Rate conviction 0-100.

YOU MUST RESPOND WITH VALID JSON ONLY.
{
    "bull_cases": [
        {
            "ticker": "PLTR",
            "conviction": 82,
            "target_profit_pct": 2.5,
            "catalysts": ["Defense contract news", "Trend score 68"],
            "pro_setup": "STOP_HUNT_LONG + EMA100_BULLISH",
            "argument": "Strong bullish case..."
        }
    ],
    "top_picks": ["PLTR", "LMT"],
    "overall_market_thesis": "Brief bull thesis for the day"
}"""

BEAR_PROMPT = """You are the BEAR RESEARCHER on a day-trading team.
Your job is to build the STRONGEST CASE AGAINST each trade.

PRO RED FLAGS TO WATCH (these are HIGH-RISK situations):
1. PARABOLIC_SHORT: 5+ green candles then first red = gravity is undefeated. Short signal.
2. RSI_EXHAUSTION_SHORT: RSI 90+ with reversal candle = about to drop. Don't buy this.
3. EMA100_BEARISH: Price BELOW 100 EMA = don't fight the trend. Any long is counter-trend.
4. STOP_HUNT_SHORT: Price spiked above prev high but fell back = bull trap.
5. MM_REFILL_DISTRIBUTION: Volume spike, no price move, slight decline = big player distributing.
6. ABOVE_VWAP too far (>2%): Overextended. Mean reversion coming.
7. "9:45 AM reversal": Retail trades like morons first 15 min. Don't chase opening moves.
8. Textbook bull flags are seen by market makers too. Expect fake breakdowns first.

Be CRITICAL but fair. Rate RISK for each ticker: 0-100.

YOU MUST RESPOND WITH VALID JSON ONLY.
{
    "bear_cases": [
        {
            "ticker": "PLTR",
            "risk_score": 35,
            "max_downside_pct": 1.5,
            "risks": ["RSI approaching overbought", "Low volume confirmation"],
            "pro_red_flag": "PARABOLIC_SHORT setup forming",
            "argument": "The bearish counter-argument..."
        }
    ],
    "avoid_list": ["MSTR"],
    "overall_risk_assessment": "Brief bear thesis for the day"
}"""

MODERATOR_PROMPT = """You are the MODERATOR of a Bull vs Bear debate on a day-trading team.

You have heard both sides. Make the FINAL DECISION using these PRO RULES:

HYBRID SCORING CONTEXT:
The technical data uses a 100-pt hybrid score:
  - Trend Baseline (max 40 pts): RSI, MACD, EMA, Bollinger
  - Execution Trigger (max 60 pts): VWAP Z-Score, Volume Delta, Smart Bounce
Score ≥ 65 = strong buy. Score < 40 = avoid.

DECISION RULES:
1. Weigh bull conviction vs bear risk score
2. If bull conviction > 65 AND bear risk < 50 → APPROVE trade
3. If bear risk > 70 → REJECT regardless of bull case

CLASSIC PRO SETUP BONUSES (add 15 conviction points if present):
   - VWAP_SMART_BOUNCE: Smart money stepping in at VWAP
   - STOP_HUNT_LONG: Retail stops hunted below prev low, reversal coming
   - MM_REFILL_ACCUMULATION: Market maker absorbing supply

CLASSIC RED FLAG PENALTIES (add 20 risk points if present):
   - PARABOLIC_SHORT: 5+ green candles just broke — gravity incoming
   - RSI_EXHAUSTION_SHORT: RSI 90+ with reversal candle
   - EMA100_BEARISH: Don't fight the 100 EMA trend

QUANT SIGNAL RULES (NEW — apply on top of classic):
   - VWAP_OVERBOUGHT (Z > 2.5): ADD 25 risk pts. Do NOT approve buy. Mean-reversion short zone.
   - VWAP_OVERSOLD  (Z < -2.5): ADD 20 conviction. Prime mean-reversion long zone.
   - SMART_BOUNCE: ADD 20 conviction. Institutional micro-structure entry confirmed.
   - VOLUME_BULLISH (delta_ratio > 0.55): ADD 15 conviction. Institutional buying detected.
   - VOLUME_BEARISH (delta_ratio < 0.45): ADD 15 risk pts. Selling pressure dominant.
   - VWAP_NEUTRAL (|Z| < 1.0): Safe zone for trend continuation plays only.

MATHEMATICAL QUANT RULES (FROM QUANT ANALYST):
You will be provided with raw mathematical data for tickers. Use it as the ultimate source of truth.
1. XGBoost Probability: If `prob_up` > 0.60, the statistical edge is LONG. Add 10 conviction points.
2. XGBoost Probability: If `prob_up` < 0.40, the statistical edge is SHORT. Add 15 risk points.
3. Factor Attribution (Alpha): If `alpha` > 0.10, the stock has genuine idiosyncratic strength (not just drifting on the SPY). Favor these for breakouts.
4. Factor Attribution (Beta): If `beta` > 1.5, the stock is highly correlated to SPY. Reject if SPY is dropping.

STOP NOTE: Do NOT specify a stop %. The TraderAgent sets ATR trailing stops dynamically.
Set stop_loss to 0 — the TraderAgent owns stop placement.

PORTFOLIO DIVERSIFICATION MANDATE (CRITICAL — MUST FOLLOW):
You MUST spread decisions across AT LEAST 2 different sectors per batch.
- If recent cycles were dominated by energy/defense (OXY, XOM, LMT, RTX), you MUST include picks from OTHER sectors.
- Required tier allocation:
    * MEGA-CAP TIER (60% buying power): TSLA, META, NVDA, AMD, GOOGL, MSFT, AAPL
      Hold-and-manage. Large cap, strong trend. Don't re-buy if already held.
    * DAY-TRADE CATALYST TIER (40% buying power): Pick from the following sub-categories each cycle:
        - Small-cap momentum: IONQ, ACHR, SOUN, RKLB, HIMS, PLUG, MARA, RIOT — high volume, high % movers
        - Mid-cap catalyst: PLTR, CRWD, PANW, DKNG, AFRM — earnings/news-driven
        - Biotech/FDA: Any biotech with a near-term catalyst (PDUFA date, trial result, approval)
        - Crypto-adjacent: MSTR, COIN, MARA — follow BTC/ETH moves intraday
        - Geopolitical: OXY, XOM, LMT, RTX — ONLY if there's a NEW event not seen in prior cycles

ANTI-REPEAT RULE: If the MEMORY shows the same tickers (OXY, XOM, LMT, RTX) appeared in the last 2+ cycles,
REJECT those for this cycle and pick from a DIFFERENT sector instead. Diversification is better than repeating the same Iran play.

CATALYST PRIORITY: Always check NEWS DATA for:
1. Earnings beat/miss announced today → instantly trade the reporting ticker
2. FDA approval/rejection → trade the specific biotech
3. Contract/partnership announcement → trade the specific company
4. Short squeeze setup (volume 5x+ avg) → research and add to day-trade tier

MEMORY INTEGRATION: Use the MEMORY section to avoid repeating decisions.

YOU MUST RESPOND WITH VALID JSON ONLY.
{
    "decisions": [
        {
            "ticker": "PLTR",
            "action": "buy",
            "conviction": 75,
            "tier": "day_trade",
            "sector": "cybersecurity",
            "position_size": "medium",
            "entry_price": 96.50,
            "target_price": 98.80,
            "stop_loss": 0,
            "reasoning": "VWAP_OVERSOLD Z=-2.8, VOLUME_BULLISH δ=0.61, SMART_BOUNCE confirmed. EMA100 bullish."
        }
    ],
    "sectors_covered": ["cybersecurity", "biotech"],
    "overall_strategy": "Specific tickers and why — not generic text",
    "market_stance": "cautiously_bullish",
    "rest_recommendation": false,
    "research_notes": "Key NEW observations for next cycle — be specific, reference tickers and sectors"
}

action values: "buy", "sell", "hold", "avoid"
tier values: "mega_cap", "day_trade"
position_size values: "small", "medium", "large"
market_stance values: "aggressive_bull", "cautiously_bullish", "neutral", "cautiously_bearish", "defensive"
"""


class ResearchTeam:
    """
    Bull/Bear research debate team. Critically assesses analyst insights
    through structured debate and produces final trade decisions.
    """

    def __init__(self):
        self.llm = get_llm()
        self.memory = get_memory()
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self._consensus_reached = False
        self._rested_last_cycle = False   # True after we skipped once — resets flag
        self._last_positions: set = set() # Track position changes to force re-debate

    def debate(
        self,
        candidates: list[str],
        technical_data: dict,
        quant_data: dict,
        fundamentals: dict,
        news_data: dict,
        sentiment_data: dict,
        positions: list[dict],
        current_cycle: int,
    ) -> dict:
        """
        Run Bull/Bear debate on candidate tickers.
        Returns final trade decisions after structured debate.
        """
        if not candidates:
            return self._empty_result("No candidates to debate")

        # Build shared context for both researchers
        context = self._build_context(candidates, technical_data, quant_data, fundamentals,
                                       news_data, sentiment_data, positions, current_cycle)

        # Get memory context
        memory_context = self.memory.get_context_for_llm(current_cycle)

        # Step 1 & 2: Run Bull and Bear researchers CONCURRENTLY
        logger.info(f"[RESEARCH] Bull and Bear researchers analyzing {len(candidates)} tickers concurrently...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bull_future = executor.submit(self._run_bull, context, memory_context)
            bear_future = executor.submit(self._run_bear, context, {"placeholder": "running concurrently"}, memory_context)
            
            bull_result = bull_future.result()
            bear_result = bear_future.result()

        # Step 3: Moderator decides
        logger.info(f"[RESEARCH] Moderator rendering decision...")
        decision = self._run_moderator(bull_result, bear_result, context, memory_context)

        # Log the full debate
        self._write_log(bull_result, bear_result, decision, current_cycle)

        # Record a SPECIFIC cycle summary (tickers + actions, not generic text)
        decisions_made = decision.get("decisions", [])
        if decisions_made:
            buys = [d["ticker"] for d in decisions_made if d.get("action") == "buy"]
            sells = [d["ticker"] for d in decisions_made if d.get("action") == "sell"]
            stance = decision.get("market_stance", "neutral")
            notes = decision.get("research_notes", "")[:150]
            summary_parts = [f"Stance={stance}"]
            if buys:
                summary_parts.append(f"BUY: {', '.join(buys)}")
            if sells:
                summary_parts.append(f"SELL: {', '.join(sells)}")
            if notes:
                summary_parts.append(notes)
            cycle_summary = " | ".join(summary_parts)
        else:
            cycle_summary = f"No trades. {decision.get('overall_strategy', '')[:150]}"
        sectors_covered = decision.get("sectors_covered", [])
        self.memory.record_cycle_summary(current_cycle, cycle_summary, sectors=sectors_covered)

        # Check if team can rest
        self._consensus_reached = decision.get("rest_recommendation", False)

        logger.info(
            f"Research Team: {len(decision.get('decisions', []))} decisions, "
            f"stance={decision.get('market_stance', 'unknown')}"
        )
        return decision

    def should_rest(self) -> bool:
        """Skip at most ONE cycle after a rest_recommendation. Then always re-debate."""
        if not self._consensus_reached:
            return False
        if self._rested_last_cycle:
            # We already rested once — force a fresh debate this cycle
            self._consensus_reached = False
            self._rested_last_cycle = False
            logger.info("[RESEARCH] Resuming debate after 1-cycle rest")
            return False
        # First rest cycle — skip this one and remember we rested
        self._rested_last_cycle = True
        logger.info("[RESEARCH] Resting for 1 cycle (consensus from last run)")
        return True

    def reset_if_positions_changed(self, current_positions: list):
        """Force the team back to debate if our holdings changed since last cycle."""
        current_set = {p.get("ticker", "") for p in current_positions}
        if current_set != self._last_positions:
            self._consensus_reached = False
            self._rested_last_cycle = False
            self._last_positions = current_set

    def research_while_closed(self, news_data: dict, positions: list[dict], current_cycle: int) -> dict:
        """Lighter research for when market is closed — prep for next day."""
        memory_context = self.memory.get_context_for_llm(current_cycle)

        prompt = f"""The market is CLOSED. Prepare research for the next trading session.

CURRENT POSITIONS: {json.dumps([{"ticker": p.get("ticker"), "pl_pct": p.get("unrealized_pl_pct", 0)} for p in positions], indent=2)}

NEWS SUMMARY: {json.dumps(news_data.get("events", [])[:5], indent=2, default=str)}

MEMORY: {memory_context}

Analyze:
1. What are the key themes for tomorrow?
2. Which tickers should we watch at market open?
3. Any overnight risks to current positions?
4. Pre-market catalysts to monitor?

Respond with JSON: {{"watch_list": [], "overnight_risks": [], "tomorrow_themes": [], "prep_notes": ""}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a market research analyst preparing for the next trading session. Respond with valid JSON only."),
                HumanMessage(content=prompt),
            ])
            result = self._parse(response.content)
            self._write_log({}, {}, {"prep_research": result}, current_cycle)
            logger.info("Research Team: prepared overnight analysis")
            return result
        except Exception as e:
            logger.error(f"Overnight research failed: {e}")
            return {"watch_list": [], "overnight_risks": [], "tomorrow_themes": [], "prep_notes": str(e)}

    def _build_context(self, candidates, tech, quant, fund, news, sent, positions, cycle) -> str:
        parts = [f"CANDIDATE TICKERS: {', '.join(candidates[:15])}"]

        # Technical summary + pro signals
        tech_summary = {}
        for t in candidates[:15]:
            if t in tech:
                pro = tech[t].get("pro_signals", {})
                tech_summary[t] = {
                    "trend_score": tech[t].get("score", 50),
                    "rsi": tech[t].get("rsi", 50),
                    "signals": tech[t].get("signals", []),
                    "entry_zone": tech[t].get("entry_zone", 0),
                    "exit_zone": tech[t].get("exit_zone", 0),
                    "current_price": tech[t].get("current_price", 0),
                    "ema100_bias": pro.get("trend_bias", {}).get("trend_bias", "unknown"),
                    "vwap_distance": float(pro.get("vwap", {}).get("distance_pct", 0)),
                    "smart_bounce": bool(pro.get("vwap", {}).get("smart_bounce", False)),
                    "stop_hunt": {k: float(v) if isinstance(v, (int, float)) else v for k, v in pro.get("stop_hunt", {}).items() if "stop_hunt" in k},
                    "mm_refill": pro.get("mm_refill", {}).get("direction_hint", "none"),
                    "parabolic": pro.get("parabolic", {}).get("direction", "none"),
                }
        # Use default=str to catch any remaining stubborn numpy types
        parts.append(f"\nTECHNICAL DATA (includes pro signals):\n{json.dumps(tech_summary, indent=2, default=str)}")

        # Quant Data (XGBoost + Factors + GBM)
        quant_summary = {}
        for t in candidates[:15]:
             if t in quant.get("quant_data", {}):
                  quant_summary[t] = quant["quant_data"][t]
        if quant_summary:
             parts.append(f"\nQUANTITATIVE MATH DATA (XGBoost, Alpha/Beta, GBM Risk Bounds):\n{json.dumps(quant_summary, indent=2, default=str)}")

        # Fundamentals summary
        fund_list = fund.get("fundamentals", [])
        if fund_list:
            parts.append(f"\nFUNDAMENTALS:\n{json.dumps(fund_list[:8], indent=2, default=str)}")

        # News events
        events = news.get("events", [])[:5]
        if events:
            parts.append(f"\nKEY NEWS:\n{json.dumps(events, indent=2, default=str)}")

        # Sentiment
        sents = sent.get("ticker_sentiments", {})
        if sents:
            parts.append(f"\nSENTIMENT:\n{json.dumps(sents, indent=2, default=str)}")

        # Current positions
        if positions:
            parts.append(f"\nCURRENT POSITIONS:\n{json.dumps([{'ticker': p.get('ticker'), 'pl_pct': p.get('unrealized_pl_pct', 0)} for p in positions], indent=2, default=str)}")

        return "\n".join(parts)

    def _run_bull(self, context: str, memory: str) -> dict:
        prompt = f"""ANALYST DATA:\n{context}\n\nMEMORY:\n{memory}\n\nBuild the strongest BULL case for each candidate ticker."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=BULL_PROMPT),
                HumanMessage(content=prompt),
            ])
            return self._parse(response.content)
        except Exception as e:
            logger.error(f"Bull researcher failed: {e}")
            return {"bull_cases": [], "top_picks": [], "overall_market_thesis": str(e)}

    def _run_bear(self, context: str, bull_result: dict, memory: str) -> dict:
        prompt = f"""ANALYST DATA:\n{context}\n\nMEMORY:\n{memory}\n\nChallenge the bull case for the candidates. Find the risks."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=BEAR_PROMPT),
                HumanMessage(content=prompt),
            ])
            return self._parse(response.content)
        except Exception as e:
            logger.error(f"Bear researcher failed: {e}")
            return {"bear_cases": [], "avoid_list": [], "overall_risk_assessment": str(e)}

    def _run_moderator(self, bull: dict, bear: dict, context: str, memory: str) -> dict:
        prompt = f"""ANALYST DATA:\n{context}\n\nBULL CASES:\n{json.dumps(bull, indent=2, default=str)}\n\nBEAR CASES:\n{json.dumps(bear, indent=2, default=str)}\n\nMEMORY:\n{memory}\n\nMake your final decisions. Weigh both sides carefully."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=MODERATOR_PROMPT),
                HumanMessage(content=prompt),
            ])
            return self._parse(response.content)
        except Exception as e:
            logger.error(f"Moderator failed: {e}")
            return self._empty_result(str(e))

    def _write_log(self, bull: dict, bear: dict, decision: dict, cycle: int):
        try:
            ts = datetime.now(timezone.utc).isoformat()
            decisions_detail = [
                {
                    "ticker": d.get("ticker", ""),
                    "action": d.get("action", ""),
                    "tier": d.get("tier", ""),
                    "sector": d.get("sector", ""),
                    "conviction": d.get("conviction", 0),
                }
                for d in decision.get("decisions", [])
            ]
            log_entry = {
                "_cycle_header": f"=== CYCLE {cycle:03d} | {ts} ===",
                "timestamp": ts,
                "cycle": cycle,
                "bull_top_picks": bull.get("top_picks", []),
                "bear_avoid_list": bear.get("avoid_list", []),
                "decisions": decisions_detail,
                "decisions_summary": [d["ticker"] + ":" + d["action"] for d in decisions_detail],
                "sectors_covered": decision.get("sectors_covered", []),
                "market_stance": decision.get("market_stance", "unknown"),
                "strategy": decision.get("overall_strategy", "")[:300],
                "research_notes": decision.get("research_notes", "")[:200],
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write research log: {e}")


    def _empty_result(self, reason: str) -> dict:
        return {
            "decisions": [],
            "overall_strategy": reason,
            "market_stance": "neutral",
            "rest_recommendation": False,
            "research_notes": "",
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
            return self._empty_result("JSON parse error")
