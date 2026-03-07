"""
War-Room Bot -- Workflow (Day Trading Architecture)
Dual-speed pipeline: fast analysts (2 min) + slow analysts (5 min) + Research Team debate.

Flow:
  [Ingest] -> [Technical + Fundamentals] -> [Research Debate] -> [Risk] -> {Execute|Halt}
              [News + Sentiment] (every 5 min, cached between fast cycles)
"""

from langgraph.graph import StateGraph, END
from graph.state import WarRoomState
from agents.technical_analyst import TechnicalAnalyst
from agents.fundamentals_analyst import FundamentalsAnalyst
from agents.news_analyst import NewsAnalyst
from agents.sentiment_analyst import SentimentAnalyst
from agents.quant_agent import quant_analyst
from agents.research_team import ResearchTeam
from agents.trader_agent import TraderAgent
from agents.risk_agent import RiskManagerAgent
from services.market_service import MarketService
from services.vector_service import VectorService
from services.broker_service import BrokerService
from utils.logger import get_logger
from utils.memory import get_memory
from config import ALL_DAY_TRADE_TICKERS, RISK_CONFIG
from datetime import datetime, timezone
import time
import json
import os

logger = get_logger("workflow")

# Initialize services
market_service = MarketService()
vector_service = VectorService()
broker_service = BrokerService()

# Initialize agents
technical_analyst = TechnicalAnalyst()
fundamentals_analyst = FundamentalsAnalyst()
news_analyst = NewsAnalyst()
sentiment_analyst = SentimentAnalyst()
research_team = ResearchTeam()
trader_agent = TraderAgent()
risk_agent = RiskManagerAgent()

# Cache for slow-loop data (persists between fast cycles)
_slow_cache = {
    "news_data": {},
    "sentiment_data": {},
    "last_slow_cycle": 0,
}


# ============================================
# Node Functions
# ============================================

def get_tracked_tickers() -> list:
    """Reads persisted tickers from fundamental_profiles.json. Falls back to ALL_DAY_TRADE_TICKERS if empty."""
    profile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fundamental_profiles.json")
    try:
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            if profiles:
                return list(profiles.keys())
    except Exception as e:
        logger.warning(f"Failed to read fundamental_profiles.json for tracked tickers: {e}")
    
    return list(ALL_DAY_TRADE_TICKERS)


def ingest_data(state: WarRoomState) -> dict:
    """Step 1: Gather portfolio state and VIX."""
    logger.info("[INGEST] Gathering market data...")

    regime_data = market_service.detect_risk_regime()
    vix_level = regime_data.get("vix_level", -1)
    portfolio = broker_service.get_account()
    positions = broker_service.get_positions()
    cycle = state.get("cycle_number", 0)

    # Sync execution_memory against real broker positions every cycle
    get_memory().sync_positions(positions, cycle)

    # Build candidate tickers: tracked tickers + any from news discovery
    candidates = get_tracked_tickers()
    niche = _slow_cache.get("news_data", {}).get("niche_tickers_discovered", [])
    for t in niche:
        if t not in candidates:
            candidates.append(t)

    # Also add held position tickers
    for pos in positions:
        t = pos.get("ticker", "")
        if t and t not in candidates:
            candidates.append(t)

    return {
        "portfolio": portfolio,
        "positions": positions,
        "vix_level": vix_level,
        "candidate_tickers": candidates,
        "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_slow_analysts(state: WarRoomState) -> dict:
    """Step 2a: News + Sentiment (every 5 min, cached between fast cycles)."""
    cycle = state.get("cycle_number", 0)

    # Run news/sentiment every 3 cycles to save LLM tokens and API calls
    if cycle % 3 != 0 and _slow_cache.get("news_data"):
        logger.info("[SLOW] Skipping LLM analysts this cycle (saving tokens)")
        return {
            "news_data": _slow_cache["news_data"],
            "sentiment_data": _slow_cache["sentiment_data"],
            "news_items": _slow_cache["news_data"].get("news_items", []),
        }

    logger.info("[SLOW] Running News + Sentiment analysts...")

    # News Analyst
    news_data = news_analyst.analyze()
    time.sleep(4)  # Rate limit: 4s between LLM calls

    # Sentiment Analyst
    candidates = state.get("candidate_tickers", ALL_DAY_TRADE_TICKERS)
    news_events = news_data.get("events", [])
    sentiment_data = sentiment_analyst.analyze(news_events, candidates)

    # Update cache
    _slow_cache["news_data"] = news_data
    _slow_cache["sentiment_data"] = sentiment_data
    _slow_cache["last_slow_cycle"] = cycle

    return {
        "news_data": news_data,
        "sentiment_data": sentiment_data,
        "news_items": news_data.get("news_items", []),
    }


def run_fast_analysts(state: WarRoomState) -> dict:
    """Step 2b: Technical + Fundamentals (every cycle, 2 min)."""
    logger.info("[FAST] Running Technical + Fundamentals analysts...")

    candidates = state.get("candidate_tickers", ALL_DAY_TRADE_TICKERS)

    # Technical Analyst (pure math, no LLM) - Runs EVERY cycle
    positions = state.get("positions", [])
    technical_data = technical_analyst.analyze(candidates, positions)

    # Quant Analyst (mathematical modeling, XGBoost, GBM, Factors)
    logger.info("[FAST] Running Quantitative Analyst...")
    quant_data = quant_analyst.analyze(candidates)

    # Fundamentals Analyst (uses LLM) - Runs every 3 cycles
    cycle = state.get("cycle_number", 0)
    if cycle % 3 != 0 and _slow_cache.get("fundamentals_data"):
        logger.info("[FAST] Skipping Fundamentals LLM this cycle")
        fundamentals_data = _slow_cache["fundamentals_data"]
    else:
        logger.info("[FAST] Running Fundamentals LLM...")
        time.sleep(3)  # Rate limit: 3s between LLM calls
        portfolio = state.get("portfolio", {})
        positions = state.get("positions", [])
        fundamentals_data = fundamentals_analyst.analyze(candidates, technical_data, portfolio, positions)
        _slow_cache["fundamentals_data"] = fundamentals_data

    return {
        "technical_data": technical_data,
        "quant_data": quant_data,
        "fundamentals_data": fundamentals_data,
    }


def run_research_debate(state: WarRoomState) -> dict:
    """Step 3: Research Team debate (Bull vs Bear)."""
    logger.info("[RESEARCH] Starting Bull/Bear debate...")

    positions = state.get("positions", [])

    # Identify if we should skip the research cycle
    cycle = state.get("cycle_number", 0)
    
    # Reset rest flag if our holdings changed (entry/exit happened last cycle)
    research_team.reset_if_positions_changed(positions)

    # Skip if it's not a research cycle (only run every 3 cycles)
    if cycle % 3 != 0:
        logger.info("[RESEARCH] Skipping debate this cycle (3-cycle token saving loop)")
        return {
            "research_decisions": [],  # Empty triggers TraderAgent to rely purely on memory for exits
            "market_stance": state.get("market_stance", "neutral"),
            "overall_strategy": state.get("overall_strategy", "Skipping research cycle") + " (Cached)",
            "research_notes": state.get("research_notes", "Skipping research cycle") + " (Cached)",
        }

    # If research team has high consensus, skip this LLM call as well
    if research_team.should_rest():
        logger.info("[RESEARCH] Team resting — consensus reached")
        return {
            "research_decisions": [],
            "market_stance": "consensus",
            "overall_strategy": "Maintaining previous strategy",
            "research_notes": "Research team resting",
        }

    technical_data = state.get("technical_data", {})
    quant_data = state.get("quant_data", {})
    fundamentals_data = state.get("fundamentals_data", {})
    news_data = state.get("news_data", {})
    sentiment_data = state.get("sentiment_data", {})
    cycle = state.get("cycle_number", 0)

    # Build candidate list from all analyst outputs
    candidates = set()

    # From technical analysis: tickers with strong signals
    for ticker, data in technical_data.items():
        signals = data.get("signals", [])
        if any(s in signals for s in ["STRONG_BUY", "BUY", "OVERSOLD", "VOLUME_SPIKE"]):
            candidates.add(ticker)

    # From fundamentals: tickers with good setups
    for f in fundamentals_data.get("fundamentals", []):
        if f.get("timing") in ["buy_now", "wait_for_dip"] and f.get("confidence", 0) > 50:
            candidates.add(f.get("ticker", ""))

    # From news: niche tickers discovered
    for t in news_data.get("niche_tickers_discovered", []):
        candidates.add(t)

    # From sentiment: strong sentiment tickers
    for ticker, sent in sentiment_data.get("ticker_sentiments", {}).items():
        if isinstance(sent, dict) and sent.get("score", 0.5) > 0.65:
            candidates.add(ticker)

    # Always include held positions for exit analysis
    for pos in positions:
        candidates.add(pos.get("ticker", ""))

    candidates.discard("")

    if not candidates:
        logger.info("[RESEARCH] No candidates after filtering")
        return {
            "research_decisions": [],
            "market_stance": "neutral",
            "overall_strategy": "No actionable opportunities found",
            "research_notes": "",
        }

    time.sleep(2)  # Rate limit pause before debate

    result = research_team.debate(
        list(candidates)[:15],
        technical_data,
        quant_data,
        fundamentals_data,
        news_data,
        sentiment_data,
        positions,
        cycle,
    )

    return {
        "research_decisions": result.get("decisions", []),
        "market_stance": result.get("market_stance", "neutral"),
        "overall_strategy": result.get("overall_strategy", ""),
        "research_notes": result.get("research_notes", ""),
    }


def run_trader_agent(state: WarRoomState) -> dict:
    """Step 4: Trader agent — holds entries until technicals align, manages dynamic exits."""
    logger.info("[TRADER] Evaluating execution timing and exits...")
    
    decisions = state.get("research_decisions", [])
    technicals = state.get("technical_data", {})
    positions = state.get("positions", [])
    cycle = state.get("cycle_number", 0)

    result = trader_agent.evaluate(decisions, technicals, positions, cycle)
    
    return result


def run_risk_manager(state: WarRoomState) -> dict:
    """Step 5: Risk management — converts trader signals into sized trades."""
    logger.info("[RISK] Evaluating trader signals...")

    decisions = state.get("trader_signals", [])
    portfolio = state.get("portfolio", {})
    positions = state.get("positions", [])
    vix_level = state.get("vix_level", -1)
    technical_data = state.get("technical_data", {})
    quant_data = state.get("quant_data", {})

    # Convert trader signals to trade signals format for risk agent
    trade_signals = []
    exit_signals = []

    for dec in decisions:
        ticker = dec.get("ticker", "")
        action = dec.get("action", "")
        conviction = dec.get("conviction", 0)

        if action == "buy":
            # Get price + quant data from technical data
            tech = technical_data.get(ticker, {})
            quant = quant_data.get("quant_data", {}).get(ticker, {})
            price = tech.get("current_price", dec.get("entry_price", 100))
            trend_score = tech.get("score", 50)
            sharpe = tech.get("sharpe", 0)
            # ATR-derived stop from TraderAgent signal (0 if not set)
            atr_5m = dec.get("atr_5m", tech.get("atr_5m", 0.0))
            
            # Allow GBM boundaries to override ATR stops if available and tighter
            stop_from_atr = dec.get("stop_loss", 0) or (
                round(price - 1.5 * atr_5m, 2) if atr_5m > 0 else round(price * 0.985, 2)
            )
            
            gbm_stop = quant.get("gbm", {}).get("stop_loss_bound_p05")
            if gbm_stop and gbm_stop > 0:
                 # Take the more conservative stop loss (max value) and default to ATR if not there.
                 final_stop = max(stop_from_atr, round(gbm_stop, 2))
            else:
                 final_stop = stop_from_atr

            gbm_target = quant.get("gbm", {}).get("profit_target_bound_p95")

            trade_signals.append({
                "ticker": ticker,
                "direction": "long",
                "confidence": conviction / 100.0,
                "trend_score": trend_score,
                "sharpe_ratio": sharpe,
                "current_price": price,
                "entry_price": price,
                "atr_5m": atr_5m,
                "target_price": round(gbm_target, 2) if gbm_target else dec.get("target_price", price * 1.03),
                "stop_loss": final_stop,
                "reasoning": dec.get("reasoning", ""),
            })
        elif action == "sell":
            exit_signals.append({
                "ticker": ticker,
                "reason": dec.get("reasoning", "Research team exit signal"),
                "urgency": "high" if conviction > 70 else "normal",
            })

    # Run through risk manager
    result = risk_agent.evaluate(
        trade_signals, portfolio, positions, vix_level
    )

    return {
        "approved_trades": result.get("approved_trades", []),
        "rejected_by_risk": result.get("rejected_trades", []),
        "exit_signals": exit_signals,
        "halt": result.get("halt", False),
        "hedge_mode": result.get("hedge_mode", False),
        "hedge_actions": result.get("hedge_actions", []),
        "risk_summary": result.get("summary", ""),
    }


def execute_trades(state: WarRoomState) -> dict:
    """Step 5: Execute approved BUY trades and exit signals."""
    logger.info("[EXEC] Executing trades...")

    approved_trades = state.get("approved_trades", [])
    exit_signals = state.get("exit_signals", [])
    executed = []
    errors = []
    memory = get_memory()
    current_cycle = state.get("cycle_number", 0)

    # 1. Execute EXIT signals first
    held_tickers = {p.get("ticker", "") for p in state.get("positions", [])}

    for exit_sig in exit_signals:
        ticker = exit_sig.get("ticker", "")
        if not ticker or ticker not in held_tickers:
            continue

        reason = exit_sig.get("reason", "exit signal")
        logger.info(f"[EXIT] Selling {ticker}: {reason}")
        try:
            # Get current price for P&L tracking
            pos = next((p for p in state.get("positions", []) if p.get("ticker") == ticker), {})
            exit_price = pos.get("current_price", 0)
            realized_pl_pct = pos.get("unrealized_pl_pct", 0)

            result = broker_service.close_position(ticker)
            result["exit_reason"] = reason
            executed.append(result)
            memory.record_sell(ticker, reason, current_cycle, exit_price=exit_price, realized_pl_pct=realized_pl_pct)
        except Exception as e:
            logger.error(f"Exit failed for {ticker}: {e}")
            errors.append({"ticker": ticker, "status": "failed", "error": str(e)})

    # 2. Execute BUY trades with buying power check
    for trade in approved_trades:
        ticker = trade.get("ticker", "")
        qty = trade.get("qty", 0)
        if qty <= 0:
            continue

        # Pre-flight buying power check
        try:
            acct = broker_service.get_account()
            current_bp = acct.get("buying_power", 0)
            price = trade.get("current_price", 100)
            cost = price * qty

            if cost > current_bp:
                affordable_qty = int(current_bp * 0.95 / price)
                if affordable_qty <= 0:
                    logger.warning(f"Skipping {ticker} — insufficient buying power")
                    errors.append({"ticker": ticker, "status": "skipped", "error": "No buying power"})
                    continue
                qty = affordable_qty
                logger.info(f"Reduced {ticker} qty to {qty} (BP: ${current_bp:,.0f})")
        except Exception as e:
            logger.warning(f"BP check failed for {ticker}: {e}")

        result = broker_service.execute_trade(ticker, "buy", qty)
        if result.get("status") == "failed":
            errors.append(result)
        else:
            executed.append(result)
            # Record buy with entry price + ATR stop for EOD report and trailing stop init
            reasoning = trade.get("reasoning", "Research team approved")
            entry_price = trade.get("entry_price") or result.get("filled_price") or 0
            atr_5m = trade.get("atr_5m", 0.0)
            memory.record_buy(ticker, reasoning, {}, current_cycle, entry_price=entry_price)
            logger.info(
                f"[BUY] {ticker} qty={qty} @ ${entry_price:.2f} | "
                f"ATR=${atr_5m:.3f} | stop=${trade.get('stop_loss', 0):.2f}"
            )

    # Store patterns in vector DB
    if executed and state.get("news_items"):
        try:
            headlines = [n.get("title", "") for n in state["news_items"][:3]]
            combined = " | ".join(headlines)
            reactions = {t.get("ticker", ""): 0 for t in executed}
            vector_service.store_event(combined, reactions)
        except Exception as e:
            logger.warning(f"Vector store failed: {e}")

    return {
        "executed_trades": executed,
        "execution_errors": errors,
    }


def should_execute(state: WarRoomState) -> str:
    """Decision node: execute trades or halt?"""
    if state.get("halt", False):
        logger.warning("[HALT] Risk manager halted trading")
        return "halt"

    approved = state.get("approved_trades", [])
    exits = state.get("exit_signals", [])

    if approved or exits:
        return "execute"
    return "halt"


# ============================================
# Build the DAG
# ============================================

def build_workflow():
    """Build the LangGraph DAG for the day-trading pipeline."""
    graph = StateGraph(WarRoomState)

    graph.add_node("ingest", ingest_data)
    graph.add_node("slow_analysts", run_slow_analysts)
    graph.add_node("fast_analysts", run_fast_analysts)
    graph.add_node("research", run_research_debate)
    graph.add_node("trader", run_trader_agent)
    graph.add_node("risk", run_risk_manager)
    graph.add_node("execute", execute_trades)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "slow_analysts")
    graph.add_edge("slow_analysts", "fast_analysts")
    graph.add_edge("fast_analysts", "research")
    graph.add_edge("research", "trader")
    graph.add_edge("trader", "risk")
    graph.add_conditional_edges("risk", should_execute, {
        "execute": "execute",
        "halt": END,
    })
    graph.add_edge("execute", END)

    return graph.compile()
