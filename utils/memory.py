"""
War-Room Bot -- Cycle Memory
Persists reasoning, position theses, and trade history between cycles
so the bot can make smarter decisions over time.
"""

import json
import os
from datetime import datetime, timezone
from utils.logger import get_logger

logger = get_logger("memory")

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cycle_memory.json")


class CycleMemory:
    """
    Persistent memory across trading cycles.

    Stores:
    - Position theses: WHY each position was opened
    - Trade history: recent buy/sell decisions
    - Cycle summaries: what happened each cycle
    - Hold timestamps: when each position was entered (for min hold enforcement)
    """

    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        """Load memory from disk."""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Memory load failed, starting fresh: {e}")

        return {
            "position_theses": {},   # ticker -> {thesis, entered_at, entry_cycle, sentiment_at_entry}
            "trade_history": [],     # Last N trades
            "cycle_summaries": [],   # Last N cycle summaries
            "cycle_count": 0,
        }

    def _save(self):
        """Save memory to disk."""
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")

    def record_buy(self, ticker: str, reasoning: str, sentiment: dict, cycle: int, entry_price: float = 0.0):
        """Record WHY a position was opened."""
        self.data["position_theses"][ticker] = {
            "thesis": reasoning,
            "entered_at": datetime.now(timezone.utc).isoformat(),
            "entry_cycle": cycle,
            "entry_price": entry_price,
            "sentiment_at_entry": sentiment,
        }

        self.data["trade_history"].append({
            "action": "buy",
            "ticker": ticker,
            "reasoning": reasoning,
            "cycle": cycle,
            "entry_price": entry_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep last 25 trades (reduced from 100 to prevent context bloat)
        self.data["trade_history"] = self.data["trade_history"][-25:]
        self._save()
        logger.info(f"Memory: recorded BUY thesis for {ticker} @ ${entry_price}")

    def record_sell(self, ticker: str, reason: str, cycle: int, exit_price: float = 0.0, realized_pl_pct: float = 0.0):
        """Record a position exit and remove thesis."""
        thesis = self.data["position_theses"].get(ticker, {})
        entry_price = thesis.get("entry_price", 0.0)

        self.data["trade_history"].append({
            "action": "sell",
            "ticker": ticker,
            "reasoning": reason,
            "cycle": cycle,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pl_pct": realized_pl_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep last 25 trades (reduced from 100 to prevent context bloat)
        self.data["trade_history"] = self.data["trade_history"][-25:]
        self._save()
        pl_str = f"{realized_pl_pct:+.2f}%" if realized_pl_pct else ""
        logger.info(f"Memory: recorded SELL for {ticker} @ ${exit_price} {pl_str}")

    def record_cycle_summary(self, cycle: int, summary: str, sectors: list | None = None):
        """Store a brief summary of what happened this cycle (with optional sector list)."""
        self.data["cycle_summaries"].append({
            "cycle": cycle,
            "summary": summary,
            "sectors": sectors or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.data["cycle_count"] = cycle

        # Keep last 15 summaries (reduced from 30 to prevent context bloat)
        self.data["cycle_summaries"] = self.data["cycle_summaries"][-15:]
        self._save()

    def get_position_thesis(self, ticker: str) -> dict:
        """Get the thesis for a held position."""
        return self.data["position_theses"].get(ticker, {})

    def get_cycles_held(self, ticker: str, current_cycle: int) -> int:
        """How many cycles has this position been held?"""
        thesis = self.data["position_theses"].get(ticker, {})
        entry_cycle = thesis.get("entry_cycle", current_cycle)
        return current_cycle - entry_cycle

    def get_recent_history(self, n: int = 10) -> list[dict]:
        """Get last N trade actions."""
        return self.data["trade_history"][-n:]

    def get_recent_summaries(self, n: int = 5) -> list[dict]:
        """Get last N cycle summaries."""
        return self.data["cycle_summaries"][-n:]

    def get_all_theses(self) -> dict:
        """Get all active position theses."""
        return self.data["position_theses"]

    def sync_positions(self, positions: list, current_cycle: int):
        """
        Reconcile execution_memory against real broker positions every cycle.
        - Adds stub theses for positions opened externally (e.g. TSLA held but not bought by bot)
        - Removes stale theses for positions that are no longer held
        """
        real_tickers = {p.get("ticker", "") for p in positions if p.get("ticker")}
        theses = self.data["position_theses"]
        changed = False

        # Add missing theses for real broker positions
        for ticker in real_tickers:
            if ticker and ticker not in theses:
                theses[ticker] = {
                    "thesis": "Position held (opened externally or pre-existing)",
                    "entered_at": datetime.now(timezone.utc).isoformat(),
                    "entry_cycle": current_cycle,
                    "sentiment_at_entry": {},
                }
                logger.info(f"Memory: synced external position {ticker} into theses")
                changed = True

        # Remove theses for positions no longer held
        stale = [t for t in list(theses.keys()) if t not in real_tickers]
        for ticker in stale:
            del theses[ticker]
            logger.info(f"Memory: removed stale thesis for closed position {ticker}")
            changed = True

        if changed:
            self._save()

    def get_context_for_llm(self, current_cycle: int) -> str:
        """
        Build a context string to inject into LLM prompts,
        giving the agent 'memory' of past decisions.
        """
        parts = []

        # Recent cycle summaries
        summaries = self.get_recent_summaries(5)
        if summaries:
            parts.append("RECENT CYCLE HISTORY:")
            for s in summaries:
                sector_str = f" [{', '.join(s.get('sectors', []))}]" if s.get('sectors') else ""
                parts.append(f"  Cycle {s['cycle']}{sector_str}: {s['summary']}")

        # Detect dominant sectors across last 5 cycles (for anti-repeat enforcement)
        all_sectors: dict = {}
        for s in summaries:
            for sec in s.get("sectors", []):
                all_sectors[sec] = all_sectors.get(sec, 0) + 1
        dominant = [sec for sec, count in all_sectors.items() if count >= 3]
        if dominant:
            parts.append(f"\nDOMINANT SECTORS (last 5 cycles): {', '.join(dominant)} — DIVERSIFY AWAY from these!")

        # Active position theses
        theses = self.get_all_theses()
        if theses:
            parts.append("\nACTIVE POSITION THESES (why we bought):")
            for ticker, info in theses.items():
                cycles_held = current_cycle - info.get("entry_cycle", current_cycle)
                parts.append(f"  {ticker} (held {cycles_held} cycles): {info.get('thesis', 'no thesis')}")

        # Recent trades
        history = self.get_recent_history(5)
        if history:
            parts.append("\nRECENT TRADES:")
            for h in history:
                parts.append(f"  {h['action'].upper()} {h['ticker']} (cycle {h['cycle']}): {h.get('reasoning', '')[:100]}")

        return "\n".join(parts) if parts else "No prior history."


# Singleton
_memory = None


def get_memory() -> CycleMemory:
    global _memory
    if _memory is None:
        _memory = CycleMemory()
    return _memory
