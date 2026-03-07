"""
Fundamentals Analyst Agent (Fast Loop — every 2 min)
Evaluates intraday price action vs VWAP, determines optimal buy/sell zones,
and calculates expected profit targets.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.logger import get_logger
from config import PORTFOLIO_ALLOCATION
from services.market_service import MarketService
import json
import os

logger = get_logger("fundamentals_analyst")
market_service = MarketService()

PROFILE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fundamental_profiles.json")

SYSTEM_PROMPT = """You are the Fundamentals Analyst of a day-trading bot.

Your job is to evaluate each ticker and determine the strategy:
1. **Buy Zone**: The optimal price range to enter (based on support levels, VWAP proximity, volume profile)
2. **Sell Target**: For new candidates, the price target for taking profit. For HELD positions, set this to 0 as we use dynamic ATR trailing stops.
3. **Expected Profit %**: Realistic intraday profit expectation (or current unrealized PnL if held)
4. **Timing**: Is NOW the right time to buy, or should we wait?
5. **Confidence**: 0-100 score of how confident you are in this setup
6. **Strategy**: A clear sentence describing the plan. If holding, it MUST describe the trailing stop strategy and current PnL adjustments (e.g., "Currently holding with a ratcheting ATR trailing stop, locking in gains").

Evaluate BOTH prospective candidates AND our currently held positions.

YOU MUST RESPOND WITH VALID JSON ONLY.

Response format:
{
    "fundamentals": [
        {
            "ticker": "PLTR",
            "current_position": false,
            "buy_zone": [95.50, 96.20],
            "sell_target": 0,
            "expected_profit_pct": 2.5,
            "timing": "buy_now",
            "confidence": 75,
            "strategy": "Trading below VWAP with volume, looking to enter...",
            "reasoning": "Volume pickup, support at 95.30"
        }
    ],
    "summary": "Brief overall assessment"
}

timing values: "buy_now", "wait_for_dip", "avoid", "already_extended", "hold_current_position"
"""


class FundamentalsAnalyst:
    """
    Fast-loop analyst: evaluates optimal buy/sell timing and profit targets.
    Uses LLM to interpret the technical data in a fundamentals context.
    """

    def __init__(self):
        self.llm = get_llm()

    def analyze(self, tickers: list[str], technical_data: dict, portfolio: dict = None, positions: list = None) -> dict:
        """
        Given technical profiles, determine buy zones, sell targets, timing.
        Portfolio and positions are used to limit recommendations based on buying power.
        """
        if not tickers:
            return {"fundamentals": [], "summary": "No tickers to analyze"}

        # Portfolio awareness
        max_positions = 5
        existing_positions = len(positions) if positions else 0
        slots_available = max(0, max_positions - existing_positions)
        buying_power = portfolio.get("buying_power", 999999) if portfolio else 999999
        equity = portfolio.get("equity", 100000) if portfolio else 100000
        held_tickers = {p.get("ticker", "") for p in (positions or [])}

        # Split buying power into mega-cap bucket vs day-trade bucket
        mega_cap_tickers = set(PORTFOLIO_ALLOCATION.get("mega_cap_tickers", []))
        mega_cap_bp = buying_power * PORTFOLIO_ALLOCATION.get("mega_cap_pct", 0.60)
        day_trade_bp = buying_power * PORTFOLIO_ALLOCATION.get("day_trade_pct", 0.40)

        # Classify candidates
        mega_cap_candidates = [t for t in tickers if t in mega_cap_tickers]
        day_trade_candidates = [t for t in tickers if t not in mega_cap_tickers]

        # Build concise data for LLM (include held tickers for profiling)
        ticker_data = {}
        # Always include held tickers first so they are never dropped by the :15 cap
        priority_tickers = list(held_tickers) + [t for t in tickers if t not in held_tickers]
        for ticker in priority_tickers[:20]:
            tech = technical_data.get(ticker, {})
            if not tech:
                # Still include held tickers even without technical data
                if ticker in held_tickers:
                    ticker_data[ticker] = {
                        "currently_held": True,
                        "note": "No technical data available — evaluate for hold/exit",
                    }
                continue
            ticker_data[ticker] = {
                "currently_held": ticker in held_tickers,
                "current_price": tech.get("current_price", 0),
                "trend_score": tech.get("score", 50),
                "rsi": tech.get("rsi", 50),
                "entry_zone": tech.get("entry_zone", 0),
                "exit_zone": tech.get("exit_zone", 0),
                "signals": tech.get("signals", []),
                "sharpe": round(tech.get("sharpe", 0), 2),
                "atr_5m": round(tech.get("atr_5m", 0), 3),
                "volume_spike": tech.get("volume_spike", False),
                "macd_trend": tech.get("macd", {}).get("trend", "neutral"),
                # Quant signals
                "vwap_zscore": tech.get("vwap_zscore", 0),
                "delta_ratio": tech.get("delta_ratio", 0.5),
            }

            # Optional deep dive for stocks not currently held (macro context)
            if not ticker_data[ticker]["currently_held"]:
                try:
                    daily = market_service.get_candles(ticker, resolution="D", days_back=60)
                    closes = daily.get("close", [])
                    if len(closes) > 10:
                        high_60d = max(closes)
                        low_60d = min(closes)
                        ticker_data[ticker]["macro_context"] = {
                            "60d_high": high_60d,
                            "60d_low": low_60d,
                            "pct_from_high": round(((tech.get("current_price", 0) - high_60d) / high_60d) * 100, 1)
                        }
                except Exception as e:
                    logger.debug(f"Failed to fetch macro context for {ticker}: {e}")

        if not ticker_data:
            return {"fundamentals": [], "summary": "No data available"}

        # Tell the LLM about portfolio constraints and tier allocation
        bp_warning = ""
        if slots_available == 0:
            bp_warning = "\n⚠️ MAX POSITIONS REACHED. Do NOT recommend any buy_now trades. Only suggest wait_for_dip or avoid."
        elif buying_power < 1000:
            bp_warning = f"\n⚠️ LOW BUYING POWER: ${buying_power:,.0f}. Recommend at most 1 trade with small size."
        elif slots_available <= 2:
            bp_warning = f"\n⚠️ Only {slots_available} position slot(s) available. Recommend at most {slots_available} trade(s)."

        prompt = f"""Analyze these tickers for day-trading entry/exit.

PORTFOLIO STATUS:
- Equity: ${equity:,.0f}
- Total Buying Power: ${buying_power:,.0f}
- Existing Positions: {existing_positions}/{max_positions}
- Open Slots: {slots_available}
- Already Holding: {', '.join(held_tickers) if held_tickers else 'none'}
{bp_warning}

BUYING POWER ALLOCATION (MUST FOLLOW):
- MEGA-CAP BUCKET: ${mega_cap_bp:,.0f} (60%) → For blue-chip holds: {', '.join(mega_cap_candidates) if mega_cap_candidates else 'none in list'}
  Use this bucket for NVDA, TSLA, META, AMD, GOOGL etc. These are hold-and-manage plays.
- DAY-TRADE BUCKET: ${day_trade_bp:,.0f} (40%) → For small/mid-cap catalysts: {', '.join(day_trade_candidates[:8]) if day_trade_candidates else 'none'}
  Use this bucket for high-beta small/mid-caps with earnings, FDA, contract, or momentum catalysts.
  These should be INTRADAY plays — tighter stops, faster exits.

Mandatory: recommend at least 1 day-trade candidate from the day-trade bucket if any show momentum.
Recommend ONLY your top {min(slots_available, 3)} BEST opportunities total (mix of tiers).

TECHNICAL DATA:
{json.dumps(ticker_data, indent=2)}

Determine buy zones, sell targets, and timing for each viable ticker.
For day-trade candidates, set tighter stops (1-1.5%) and faster targets (2-4%).
For mega-caps, use standard stops (1.5-2%) and targets (2-5%)."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = self._parse(response.content)
            
            fundies = result.get('fundamentals', [])
            self._update_profiles(fundies, held_tickers)
            
            logger.info(f"Fundamentals Analyst: {len(fundies)} profiles evaluated")
            return result
        except Exception as e:
            logger.error(f"Fundamentals analysis failed: {e}")
            return {"fundamentals": [], "summary": f"Error: {e}"}

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
            return {"fundamentals": [], "summary": "Parse error"}

    def _update_profiles(self, fundamentals: list, held_tickers: set):
        """Maintains individual profiles for held positions and strong prospective candidates, omitting sold ones."""
        os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
        profiles = {}
        
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception:
                pass
                
        # Upsert new fundamentals
        for fund in fundamentals:
            ticker = fund.get("ticker")
            if ticker:
                profiles[ticker] = fund

        # Fix: forcibly mark current_position=True for any broker-confirmed held ticker,
        # regardless of what the LLM returned (LLM often defaults to false)
        for ticker in held_tickers:
            if ticker in profiles:
                profiles[ticker]["current_position"] = True
            else:
                # Held ticker not analysed at all — add a stub so it's tracked
                profiles[ticker] = {
                    "ticker": ticker,
                    "current_position": True,
                    "timing": "hold_current_position",
                    "confidence": 50,
                    "strategy": "Position held — awaiting next analysis cycle.",
                    "reasoning": "Ticker not in this cycle's candidate list.",
                }
                
        # Clean up: remove explicit "avoid" tickers from profiles unless they are currently held
        final_profiles = {}
        avoid_tickers = {f.get("ticker") for f in fundamentals if f.get("timing") == "avoid"}
        
        for t, p in profiles.items():
            if t in held_tickers or t not in avoid_tickers:
                final_profiles[t] = p

        try:
            with open(PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(final_profiles, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save fundamental profiles: {e}")
