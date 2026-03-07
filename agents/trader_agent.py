"""
Trader Agent — Hybrid Execution Layer
Sits between Research and Risk.

ENTRY: Hybrid gate — requires BOTH traditional trend alignment AND quant execution confirmation.
EXIT:  Purely dynamic ATR trailing stops. NO static +3% / -2% thresholds.

ATR Trailing Stop Logic:
  On entry:  stop_price = entry_price - 1.5 × ATR_5m
  Each cycle: new_candidate = current_price - 1.5 × ATR_5m
              trailing_stop = max(trailing_stop, new_candidate)  ← RATCHETS UP ONLY
  Sell when:  current_price <= trailing_stop
  Also exit:  VWAP Z-score > 3.0 (statistical overbought mean-reversion signal)
"""

import os
import json
from datetime import datetime, timezone
from utils.logger import get_logger

logger = get_logger("trader_agent")

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "execution_memory.json")

# ATR multiplier for stop placement
ATR_STOP_MULTIPLIER  = 1.5   # stop = entry - 1.5 × ATR
ATR_TARGET_MULTIPLIER = 3.0  # target = entry + 3.0 × ATR (2:1 R:R soft guide)

# Z-score threshold to trigger mean-reversion exit
ZSCORE_EXIT_THRESHOLD = 3.0

# Minimum execution conviction from research team
MIN_CONVICTION = 50

# Max number of cycles to wait for entry setup (15 cycles × 2 min ≈ 30 min)
PENDING_ENTRY_TTL = 15


class TraderAgent:
    """
    Hybrid execution: trend-aligned entries, ATR-driven exits.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        self.memory = self._load_memory()

    def _load_memory(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {"pending_entries": {}}

    def _save_memory(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save execution memory: {e}")

    def evaluate(self, research_decisions: list, technical_data: dict, positions: list, cycle: int) -> dict:
        """
        Evaluate research decisions against live hybrid technicals.

        Returns:
            {
                "trader_signals": [...],    # sized buy/sell signals for risk agent
                "trader_strategy": str      # execution narrative logged to terminal
            }
        """
        trader_signals = []
        strategy_notes = []

        # ── STEP 1: Load research decisions into pending queue ─────────────────
        for dec in research_decisions:
            ticker = dec.get("ticker", "")
            action = dec.get("action", "")
            conviction = dec.get("conviction", 0)

            if not ticker:
                continue

            if action == "buy" and conviction >= MIN_CONVICTION:
                if ticker not in self.memory["pending_entries"]:
                    logger.info(f"Trader: Queuing {ticker} for hybrid entry check (conviction={conviction})")
                    self.memory["pending_entries"][ticker] = {
                        "decision": dec,
                        "added_cycle": cycle,
                        "expires_in": PENDING_ENTRY_TTL,
                        # ATR stop will be populated when entry fires
                        "atr_5m": 0.0,
                        "stop_price": 0.0,
                        "trailing_stop": 0.0,
                        "entry_price": 0.0,
                    }
                    strategy_notes.append(f"Queued {ticker} (conviction={conviction}) for setup confirmation.")

            elif action == "sell":
                # Immediate research-mandated exit
                trader_signals.append({
                    "ticker": ticker,
                    "action": "sell",
                    "reasoning": f"Research exit: {dec.get('reasoning', '')}",
                    "urgency": "high" if conviction > 70 else "normal",
                })

        # ── STEP 2: Evaluate pending entries — hybrid gate ─────────────────────
        expired = []
        for ticker, entry_data in self.memory["pending_entries"].items():
            if "decision" not in entry_data:
                continue

            entry_data["expires_in"] = entry_data.get("expires_in", 0) - 1
            if entry_data["expires_in"] <= 0:
                logger.info(f"Trader: {ticker} pending entry expired (no setup found in time)")
                expired.append(ticker)
                continue

            tech = technical_data.get(ticker, {})
            if not tech:
                strategy_notes.append(f"Waiting on {ticker}: no technical data.")
                continue

            # ── HYBRID ENTRY GATE ──────────────────────────────────────────────
            # TIER A — Trend Baseline (classic indicators, max 40 pts)
            raw_score = tech.get("score", 50)         # hybrid score (0-100)
            trend_score = tech.get("trend_score", 20)  # 0-40
            rsi = tech.get("rsi", 50)

            trend_aligned = (
                trend_score >= 18                 # at least slightly bullish classic signal
                and rsi < 72                      # not RSI overbought
                and "EMA100_BEARISH" not in tech.get("signals", [])  # not fighting the macro trend
            )

            # TIER B — Execution Trigger (quant metrics, max 60 pts)
            z = tech.get("vwap_zscore", 0.0)
            dr = tech.get("delta_ratio", 0.5)
            smart_bounce = tech.get("smart_bounce", False)
            price = tech.get("current_price", 0)
            atr = tech.get("atr_5m", 0.0)

            # Strongest signal: mean-reversion off VWAP extreme + volume confirmation
            is_mean_reversion_entry = (
                z < -2.0
                and dr > 0.55
                and price > 0
            )

            # Continuation signal: price near VWAP, volume bullish, trend aligned
            is_continuation_entry = (
                abs(z) < 1.0              # price close to VWAP (fair value zone)
                and dr > 0.60             # strong institutional buying
                and raw_score >= 55       # hybrid score shows bullish momentum
            )

            # Smart bounce bonus (strictest institutional pattern)
            is_smart_bounce_entry = smart_bounce and dr > 0.55 and trend_aligned

            fire_entry = (
                trend_aligned
                and (is_mean_reversion_entry or is_continuation_entry or is_smart_bounce_entry)
            )

            if fire_entry:
                # Calculate ATR-based stop on entry
                stop_price = round(price - ATR_STOP_MULTIPLIER * atr, 2) if atr > 0 else round(price * 0.985, 2)
                target_price = round(price + ATR_TARGET_MULTIPLIER * atr, 2) if atr > 0 else round(price * 1.03, 2)

                # Store stop/ATR so we can ratchet it during position holding
                entry_data["entry_price"] = price
                entry_data["atr_5m"] = atr
                entry_data["stop_price"] = stop_price
                entry_data["trailing_stop"] = stop_price  # initial trailing = initial stop

                dec = entry_data["decision"]
                entry_type = (
                    "mean_reversion" if is_mean_reversion_entry
                    else "smart_bounce" if is_smart_bounce_entry
                    else "continuation"
                )

                logger.info(
                    f"Trader: FIRE {ticker} [{entry_type}] @ ${price:.2f} | "
                    f"stop=${stop_price:.2f} | target=${target_price:.2f} | "
                    f"Z={z:+.2f} | δ={dr:.2f} | ATR=${atr:.3f}"
                )

                trader_signals.append({
                    "ticker": ticker,
                    "action": "buy",
                    "confidence": dec.get("conviction", 50) / 100.0,
                    "trend_score": raw_score,
                    "current_price": price,
                    "entry_price": price,
                    "entry_type": entry_type,
                    "target_price": dec.get("target_price", target_price),
                    "stop_loss": stop_price,
                    "atr_5m": atr,
                    "vwap_zscore": z,
                    "delta_ratio": dr,
                    "reasoning": (
                        f"[{entry_type}] Hybrid entry: trend={trend_score:.0f}/40, "
                        f"Z={z:+.2f}, δ={dr:.2f}, ATR=${atr:.3f}, "
                        f"stop=${stop_price:.2f}. "
                        f"Research: {dec.get('reasoning', '')[:100]}"
                    ),
                })
                expired.append(ticker)

            else:
                wait_reason = []
                if not trend_aligned:
                    wait_reason.append(f"trend weak ({trend_score:.0f}/40)")
                if z > 1.5:
                    wait_reason.append(f"Z={z:+.2f} overbought")
                if dr <= 0.50:
                    wait_reason.append(f"δ={dr:.2f} (no buying pressure)")
                strategy_notes.append(f"Waiting on {ticker}: {', '.join(wait_reason) or 'conditions not met'}")

        for t in expired:
            self.memory["pending_entries"].pop(t, None)

        # ── STEP 3: Monitor held positions — ATR trailing stop ─────────────────
        held = {p.get("ticker", ""): p for p in positions}

        for ticker, pos in held.items():
            tech = technical_data.get(ticker, {})
            if not tech:
                continue

            current_price = tech.get("current_price", 0) or pos.get("current_price", 0)
            atr = tech.get("atr_5m", 0.0)
            z = tech.get("vwap_zscore", 0.0)
            pl_pct = pos.get("unrealized_pl_pct", 0)

            # Retrieve or initialise trailing stop from memory
            mem = self.memory["pending_entries"].get(ticker, {})
            trailing_stop = mem.get("trailing_stop", 0.0)
            entry_price = mem.get("entry_price", pos.get("entry_price", current_price))

            # If we don't have a trailing stop recorded (e.g. position opened externally),
            # seed it from current ATR or a conservative 2% from entry
            if trailing_stop == 0.0 and current_price > 0:
                if atr > 0:
                    trailing_stop = current_price - ATR_STOP_MULTIPLIER * atr
                else:
                    trailing_stop = entry_price * 0.98
                if ticker not in self.memory["pending_entries"]:
                    self.memory["pending_entries"][ticker] = {}
                self.memory["pending_entries"][ticker]["trailing_stop"] = trailing_stop
                self.memory["pending_entries"][ticker]["entry_price"] = entry_price

            # Ratchet trailing stop UPWARD only
            if atr > 0 and current_price > 0:
                new_candidate_stop = round(current_price - ATR_STOP_MULTIPLIER * atr, 2)
                if new_candidate_stop > trailing_stop:
                    trailing_stop = new_candidate_stop
                    self.memory["pending_entries"].setdefault(ticker, {})["trailing_stop"] = trailing_stop
                    logger.debug(f"Trailing stop ratcheted up for {ticker}: ${trailing_stop:.2f}")

            # ── EXIT CONDITIONS ────────────────────────────────────────────────
            exit_reason = None

            # 1. ATR trailing stop hit (primary exit)
            if current_price > 0 and trailing_stop > 0 and current_price <= trailing_stop:
                exit_reason = f"ATR Trailing Stop hit: ${current_price:.2f} ≤ ${trailing_stop:.2f}"

            # 2. Statistical overbought (mean-reversion exit)
            # Only exit if overbought AND showing institutional selling pressure or momentum weakness
            elif z > ZSCORE_EXIT_THRESHOLD and tech.get("delta_ratio", 0.5) < 0.55:
                exit_reason = f"VWAP Z-Score overbought exit with weakness: Z={z:+.2f} > {ZSCORE_EXIT_THRESHOLD}, δ={tech.get('delta_ratio', 0.5):.2f}"

            # 3. Hawkes Process / ETAS (Distribution Clustering)
            # If volume is spiking (aftershocks) and delta_ratio is extremely low (bearish clustering),
            # this mathematically indicates institutional distribution, not just retail noise.
            elif tech.get("volume_spike", False) and tech.get("delta_ratio", 0.5) < 0.40:
                hawkes_intensity = "HIGH"
                exit_reason = f"Hawkes Distribution Clustering detected: intensity={hawkes_intensity}, δ={tech.get('delta_ratio'):.2f} + Vol Spike"

            # 4. Catastrophic technical breakdown (safety net: score collapses AND negative P&L)
            elif tech.get("score", 50) < 25 and pl_pct < -3.0:
                exit_reason = f"Technical collapse + P&L {pl_pct:.2f}% (safety net)"

            if exit_reason:
                logger.info(f"Trader: EXIT {ticker} — {exit_reason}")
                trader_signals.append({
                    "ticker": ticker,
                    "action": "sell",
                    "reasoning": f"Dynamic exit: {exit_reason}",
                    "urgency": "high",
                })
                strategy_notes.append(f"Exit {ticker}: {exit_reason}")
                # Clean up memory
                self.memory["pending_entries"].pop(ticker, None)
            else:
                if trailing_stop > 0:
                    strategy_notes.append(
                        f"Holding {ticker}: trailing stop=${trailing_stop:.2f} | "
                        f"price=${current_price:.2f} | Z={z:+.2f}"
                    )

        self._save_memory()

        if not strategy_notes:
            strategy_notes.append("No triggers met this cycle. Monitoring pipeline.")

        return {
            "trader_signals": trader_signals,
            "trader_strategy": " | ".join(strategy_notes[:6]),  # cap for terminal width
        }
