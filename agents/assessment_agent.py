"""
Assessment Agent — End-of-Day Performance Review
Runs when market closes. Evaluates day's P&L, writes detailed lesson report.
Reads report at pre-market to brief the team on yesterday's lessons.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.memory import get_memory
from utils.logger import get_logger
import json
import os
from datetime import datetime, timezone
from services.broker_service import BrokerService

logger = get_logger("assessment_agent")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reports")

EOD_SYSTEM_PROMPT = """You are the Assessment Agent for a day-trading bot. Your job is to write a BRUTALLY HONEST end-of-day report.

You receive:
- A timeline of every trade made today (buy timestamp, sell timestamp, prices, P&L)
- The research thesis that justified each buy
- Remaining open positions (mega-cap holds not closed)
- Overall P&L

Write a report that covers:
1. **P&L Summary**: Total gain/loss, win rate, best/worst trade
2. **Trade-by-Trade Analysis**: For EACH trade — was the thesis correct? Did we enter/exit at the right time? What signal did we miss?
3. **Timing Quality**: Did we buy too late? Sell too early? Chase? 
4. **Tier Performance**: How did mega-cap holds perform vs small/mid-cap day trades?
5. **Mistakes**: Specific, named mistakes (e.g. "Bought IONQ on news excitement but ignored RSI overbought at 78")
6. **Tomorrow's Plan**: Concrete watch list, avoid list, and ONE strategy adjustment

Be SPECIFIC. Name tickers. Name prices. Name mistakes. Vague feedback is useless.

YOU MUST RESPOND WITH VALID JSON ONLY.
{
    "pnl_summary": {
        "total_pnl_pct": 0.5,
        "total_pnl_dollars": 500,
        "win_rate": 0.6,
        "trades_count": 5,
        "best_trade": {"ticker": "PLTR", "pnl_pct": 2.1, "reason": "defense contract catalyst"},
        "worst_trade": {"ticker": "IONQ", "pnl_pct": -2.3, "reason": "ignored RSI overbought"}
    },
    "trade_analysis": [
        {
            "ticker": "PLTR",
            "tier": "day_trade",
            "action": "buy then sell",
            "entry_price": 96.50,
            "exit_price": 98.60,
            "pnl_pct": 2.18,
            "thesis": "Defense contract news",
            "thesis_correct": true,
            "timing_grade": "A",
            "notes": "Entry timed well after volume confirmation. Exit before full target — left 0.5% on the table."
        }
    ],
    "open_positions": [
        {
            "ticker": "TSLA",
            "tier": "mega_cap",
            "unrealized_pnl_pct": -1.2,
            "recommendation": "hold — thesis intact, EMA100 still bullish"
        }
    ],
    "timing_grade": "B",
    "strategy_grade": "B+",
    "day_trade_tier_grade": "B",
    "mega_cap_tier_grade": "B+",
    "mistakes": [
        "Bought IONQ at 10:15 on AI news — RSI was already 78, entered overbought. Stop hit in 33 min.",
        "Sold PLTR at $98.10 but target was $99.50 — exited early due to one red candle, missed 1.4%"
    ],
    "lessons": [
        "Never buy when RSI > 75 on a day-trade. Wait for pullback to VWAP.",
        "Stick to profit targets. One red candle is not a reversal signal."
    ],
    "tomorrow_plan": {
        "watch_list": ["NVDA", "ACHR"],
        "avoid_list": ["IONQ"],
        "strategy_adjustment": "Set RSI filter: no buys above RSI 72 on day-trade tier"
    },
    "full_report": "Narrative text of the full analysis..."
}
"""

PREMARKET_SYSTEM_PROMPT = """You are the War-Room Bot's Pre-Market Briefing Agent.

You have read yesterday's trading report. Your job is to prepare a concise morning briefing for the trading team BEFORE the market opens.

Generate:
1. A summary of yesterday's KEY lessons (max 3 bullet points — the most important ones)  
2. Today's watch list with brief rationale
3. Today's avoid list with reason
4. One concrete rule to apply today (e.g. "No RSI > 72 entries on day-trade tier")
5. Market context to watch for (geopolitical, news, sector rotation)

Be direct and tactical. This briefing is read RIGHT BEFORE trading starts.

YOU MUST RESPOND WITH VALID JSON ONLY.
{
    "key_lessons": ["Lesson 1", "Lesson 2", "Lesson 3"],
    "watch_list": [{"ticker": "NVDA", "reason": "Broke resistance, analyst upgrade yesterday"}],
    "avoid_list": [{"ticker": "IONQ", "reason": "Hit stop loss twice this week, overextended"}],
    "rule_of_the_day": "Wait for RSI < 65 before entering day-trade positions",
    "market_context": "Watch Fed speakers at 10am. Oil elevated — defense sector may lag.",
    "opening_bias": "cautiously_bullish"
}
"""


class AssessmentAgent:
    """End-of-day assessment: evaluates performance, writes report, extracts lessons."""

    def __init__(self):
        self.llm = get_llm()
        self.memory = get_memory()
        self.broker = BrokerService()
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def assess_day(
        self,
        portfolio: dict,
        positions: list[dict],
        eod_closed_positions: list[dict],
        current_cycle: int,
    ) -> dict:
        """
        Run end-of-day assessment and write report.
        eod_closed_positions: list returned by broker_service.close_day_trade_positions()
        """
        memory_context = self.memory.get_context_for_llm(current_cycle)
        trade_history = self.memory.get_recent_history(50)

        # Reconstruct trade timeline from memory
        timeline = self._build_trade_timeline(trade_history, eod_closed_positions)

        # Format open positions (mega-cap holds remaining)
        open_pos_str = json.dumps(
            [
                {
                    "ticker": p.get("ticker"),
                    "tier": "mega_cap",
                    "entry_price": p.get("entry_price"),
                    "current_price": p.get("current_price"),
                    "unrealized_pl_pct": round(p.get("unrealized_pl_pct", 0), 2),
                    "market_value": p.get("market_value"),
                }
                for p in positions
            ],
            indent=2,
        )

        prompt = f"""Evaluate today's trading performance. Write a thorough, specific report.

PORTFOLIO (End of Day):
{json.dumps(portfolio, indent=2)}

TODAY'S TRADE TIMELINE:
{json.dumps(timeline, indent=2)}

EOD LIQUIDATION RESULTS (day-trade positions closed at market close):
{json.dumps(eod_closed_positions, indent=2)}

REMAINING OPEN POSITIONS (mega-cap holds — NOT closed):
{open_pos_str}

STRATEGY MEMORY (recent cycle history):
{memory_context}

For each trade, evaluate: Was the thesis correct? Was timing good? What should we do differently?
Be SPECIFIC — name prices, name patterns, name mistakes."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=EOD_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = self._parse(response.content)

            self._write_report(result, timeline, eod_closed_positions)

            # Store top 3 lessons in memory
            lessons = result.get("lessons", [])
            if lessons:
                self.memory.record_cycle_summary(
                    current_cycle,
                    f"EOD Lessons: {'; '.join(lessons[:3])}"
                )

            pnl = result.get("pnl_summary", {})
            logger.info(
                f"EOD Assessment complete | Strategy: {result.get('strategy_grade','?')} | "
                f"P&L: {pnl.get('total_pnl_pct', 0):+.2f}% (${pnl.get('total_pnl_dollars', 0):+,.0f})"
            )
            return result

        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return {"pnl_summary": {}, "lessons": [str(e)], "full_report": f"Assessment error: {e}"}

    def generate_premarket_briefing(self) -> dict:
        """
        Read yesterday's report and generate a pre-market briefing.
        Called 20 minutes before market open.
        """
        report_content = self._load_latest_report_text()
        if not report_content:
            logger.info("Pre-market: No previous report found, skipping briefing")
            return {}

        prompt = f"""Read yesterday's trading report and generate today's pre-market briefing.

YESTERDAY'S REPORT:
{report_content[:4000]}

Generate a concise morning briefing: key lessons to apply today, watch/avoid lists, and a single rule of the day."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=PREMARKET_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            briefing = self._parse(response.content)

            # Print to terminal for visibility
            logger.info("=" * 60)
            logger.info("📋 PRE-MARKET BRIEFING")
            logger.info("=" * 60)
            for lesson in briefing.get("key_lessons", []):
                logger.info(f"  📌 {lesson}")
            logger.info(f"  🎯 Rule of Day: {briefing.get('rule_of_the_day', '')}")
            watch = [w.get("ticker") for w in briefing.get("watch_list", [])]
            avoid = [a.get("ticker") for a in briefing.get("avoid_list", [])]
            logger.info(f"  👀 Watch: {', '.join(watch)}")
            logger.info(f"  🚫 Avoid: {', '.join(avoid)}")
            logger.info(f"  🌐 Context: {briefing.get('market_context', '')}")
            logger.info("=" * 60)

            # Store in memory so research_team prompt gets it
            if briefing:
                self.memory.record_cycle_summary(
                    0,
                    f"Pre-Market Brief: {briefing.get('rule_of_the_day', '')} | "
                    f"Watch={','.join(watch)} | Avoid={','.join(avoid)}"
                )

            return briefing

        except Exception as e:
            logger.error(f"Pre-market briefing failed: {e}")
            return {}

    def load_yesterdays_lessons(self) -> list[str]:
        """Load lessons from the most recent report (used at startup)."""
        try:
            reports = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")])
            if not reports:
                return []
            latest = os.path.join(REPORTS_DIR, reports[-1])
            with open(latest, "r", encoding="utf-8") as f:
                content = f.read()
            if "## Lessons Learned" in content:
                lessons_section = content.split("## Lessons Learned")[1].split("##")[0]
                return [l.strip("- ").strip() for l in lessons_section.strip().split("\n") if l.strip().startswith("-")]
            return []
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _build_trade_timeline(self, trade_history: list, eod_closed: list) -> list[dict]:
        """
        Pair buy records with their sell records to build a trade timeline.
        Enrich with EOD close data where available.
        Fetch today's actual filled prices from Alpaca to overwrite $0.00 entries.
        """
        eod_by_ticker = {r.get("ticker"): r for r in eod_closed if r.get("ticker")}
        timeline = []
        buys = {}
        
        # 1. Fetch exact fills from Alpaca
        filled_orders = self.broker.get_todays_filled_orders()
        alpaca_fills = {}
        for o in filled_orders:
            key = (o['ticker'], o['side'])
            alpaca_fills.setdefault(key, []).append(o)

        # Helper to get the best matching price
        def get_exact_fill(ticker: str, side: str, fallback: float) -> float:
            fills = alpaca_fills.get((ticker, side), [])
            if fills:
                # Alpaca returns newest first. Pop(-1) yields the oldest available (FIFO)
                return fills.pop(-1)["filled_price"]
            return fallback

        # 2. Reconstruct from memory
        for entry in trade_history:
            action = entry.get("action", "")
            ticker = entry.get("ticker", "")
            ts = entry.get("timestamp", "")

            if action == "buy":
                mem_entry = entry.get("entry_price", 0)
                exact_entry = get_exact_fill(ticker, "buy", mem_entry)
                
                buys[ticker] = {
                    "ticker": ticker,
                    "buy_time": ts,
                    "entry_price": exact_entry,
                    "reasoning": entry.get("reasoning", ""),
                    "cycle": entry.get("cycle", 0),
                }
            elif action == "sell" and ticker in buys:
                buy = buys.pop(ticker)
                
                entry_price = buy.get("entry_price") or entry.get("entry_price", 0)
                if entry_price == 0:
                     entry_price = get_exact_fill(ticker, "buy", entry_price)

                mem_exit = entry.get("exit_price", 0)
                exit_price = get_exact_fill(ticker, "sell", mem_exit)
                
                pl_pct = entry.get("realized_pl_pct", 0)
                if not pl_pct and entry_price and exit_price:
                    pl_pct = ((exit_price - entry_price) / entry_price) * 100

                timeline.append({
                    "ticker": ticker,
                    "buy_time": buy["buy_time"],
                    "sell_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "realized_pl_pct": round(pl_pct, 2),
                    "result": "win" if pl_pct > 0 else "loss",
                    "thesis": buy.get("reasoning", ""),
                    "exit_reason": entry.get("reasoning", ""),
                })

        # Add any remaining buys (still open or closed at EOD)
        for ticker, buy in buys.items():
            eod = eod_by_ticker.get(ticker, {})
            pl_pct = eod.get("realized_pl_pct", 0)
            exit_price = eod.get("exit_price", 0)
            timeline.append({
                "ticker": ticker,
                "buy_time": buy["buy_time"],
                "sell_time": "EOD close" if eod else "still open",
                "entry_price": buy.get("entry_price", 0),
                "exit_price": exit_price,
                "realized_pl_pct": round(pl_pct, 2),
                "result": "win" if pl_pct > 0 else ("open" if not eod else "loss"),
                "thesis": buy.get("reasoning", ""),
                "exit_reason": "EOD mandatory close" if eod else "position still open",
            })

        return sorted(timeline, key=lambda x: x.get("buy_time", ""))

    def _write_report(self, result: dict, timeline: list, eod_closed: list):
        """Write the daily report as a markdown file with full trade table."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = os.path.join(REPORTS_DIR, f"{today}.md")

        pnl = result.get("pnl_summary", {})
        lessons = result.get("lessons", [])
        mistakes = result.get("mistakes", [])
        plan = result.get("tomorrow_plan", {})
        narrative = result.get("full_report", "No narrative.")

        # Build trade table
        trade_rows = []
        for t in timeline:
            pl = t.get("realized_pl_pct", 0)
            pl_str = f"{pl:+.2f}%" if pl else "open"
            win_loss = "✅" if pl > 0 else ("⏳" if pl == 0 else "❌")
            trade_rows.append(
                f"| {t.get('ticker','?')} | ${t.get('entry_price',0):.2f} | "
                f"${t.get('exit_price',0):.2f} | {pl_str} | {win_loss} | "
                f"{t.get('buy_time','?')[:19]} | {t.get('sell_time','?')[:19]} | "
                f"{t.get('thesis','')[:60]} |"
            )
        trade_table = "\n".join(trade_rows) if trade_rows else "_No trades recorded_"

        # EOD close summary
        eod_rows = []
        for e in eod_closed:
            if e.get("status") == "closed":
                pl = e.get("realized_pl_pct", 0)
                eod_rows.append(f"- **{e['ticker']}**: Closed @ ${e.get('exit_price',0):.2f} | {pl:+.2f}% (${e.get('realized_pl_dollars',0):+.0f})")
            else:
                eod_rows.append(f"- **{e['ticker']}**: Close FAILED — {e.get('error','unknown error')}")
        eod_section = "\n".join(eod_rows) if eod_rows else "_No EOD liquidation required_"

        report = f"""# 📊 Trading Report — {today}

## P&L Summary
| Metric | Value |
|--------|-------|
| Total P&L | {pnl.get('total_pnl_pct',0):+.2f}% (${pnl.get('total_pnl_dollars',0):+,.0f}) |
| Win Rate | {pnl.get('win_rate',0):.0%} |
| Trades | {pnl.get('trades_count',0)} |
| Best Trade | {pnl.get('best_trade',{}).get('ticker','?')} ({pnl.get('best_trade',{}).get('pnl_pct',0):+.1f}%) |
| Worst Trade | {pnl.get('worst_trade',{}).get('ticker','?')} ({pnl.get('worst_trade',{}).get('pnl_pct',0):+.1f}%) |

## Grades
| Dimension | Grade |
|-----------|-------|
| Timing | {result.get('timing_grade','?')} |
| Strategy (Overall) | {result.get('strategy_grade','?')} |
| Mega-Cap Tier | {result.get('mega_cap_tier_grade','?')} |
| Day-Trade Tier | {result.get('day_trade_tier_grade','?')} |

## Trade Timeline
| Ticker | Entry | Exit | P&L% | W/L | Buy Time | Sell Time | Thesis |
|--------|-------|------|------|-----|----------|-----------|--------|
{trade_table}

## EOD Liquidation (Day-Trade Positions Closed at 3:57pm ET)
{eod_section}

## Mistakes Made Today
{chr(10).join(f'- {m}' for m in mistakes) if mistakes else '_None identified_'}

## Lessons Learned
{chr(10).join(f'- {l}' for l in lessons) if lessons else '_No lessons recorded_'}

## Tomorrow's Plan
- **Watch**: {', '.join(plan.get('watch_list', []))}
- **Avoid**: {', '.join(plan.get('avoid_list', []))}
- **Strategy Adjustment**: {plan.get('strategy_adjustment', 'None')}

## Full Analysis
{narrative}
"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"📝 EOD report written → {filepath}")
        except Exception as e:
            logger.error(f"Failed to write report: {e}")

    def _load_latest_report_text(self) -> str:
        """Load the most recent daily report as raw text."""
        try:
            reports = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")])
            if not reports:
                return ""
            latest = os.path.join(REPORTS_DIR, reports[-1])
            with open(latest, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

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
            return {"pnl_summary": {}, "lessons": ["Parse error"], "full_report": raw[:500]}
