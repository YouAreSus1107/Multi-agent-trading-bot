"""
War-Room Bot -- Risk Manager Agent
Equal-weight allocation, no position limit, buying power aware.
"""

from config import RISK_CONFIG
from utils.indicators import kelly_criterion
from utils.logger import get_logger, log_kill_switch

logger = get_logger("risk_agent")


class RiskManagerAgent:
    """
    Risk Manager Agent -- Equal Weight, No Position Limit.

    Rules:
    1. VIX > kill_switch -> hedge mode
    2. Equal-weight: divide buying power evenly across trades in a cycle
    3. Each trade capped at max_position_pct of equity
    4. Daily loss halt
    5. No hard position count limit
    """

    def __init__(self):
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.halted = False
        self.hedge_mode = False

    def evaluate(
        self,
        trade_signals: list[dict],
        portfolio: dict,
        positions: list[dict],
        vix_level: float,
    ) -> dict:
        equity = portfolio.get("equity", 0)
        buying_power = portfolio.get("buying_power", 0)

        if equity <= 0:
            return self._halt_response("Account equity is zero or negative")

        # CHECK 1: Daily Loss Kill Switch
        daily_loss_pct = self._calculate_daily_loss(positions, equity)
        max_loss = RISK_CONFIG["max_daily_loss_pct"]

        if daily_loss_pct >= max_loss:
            log_kill_switch(logger, "MAX_DAILY_LOSS", daily_loss_pct=daily_loss_pct)
            self.halted = True
            return self._halt_response(
                f"Daily loss ({daily_loss_pct:.1%}) exceeds maximum ({max_loss:.1%}). HALTED."
            )

        # CHECK 2: VIX Kill Switch
        vix_threshold = RISK_CONFIG["vix_kill_switch"]
        if vix_level > 0 and vix_level >= vix_threshold:
            log_kill_switch(logger, "VIX_SPIKE", vix_level=vix_level)
            self.hedge_mode = True
            hedge_actions = self._generate_hedge_actions(positions, equity)
            return {
                "approved_trades": [],
                "rejected_trades": [
                    {"ticker": s.get("ticker", ""), "reason": f"VIX {vix_level} > {vix_threshold}"}
                    for s in trade_signals
                ],
                "halt": False,
                "hedge_mode": True,
                "hedge_actions": hedge_actions,
                "risk_summary": f"HEDGE MODE: VIX at {vix_level}",
            }

        # FILTER: remove already-held tickers and invalid signals
        held_tickers = {p.get("ticker", "") for p in positions}
        valid_signals = []
        rejected = []

        for signal in trade_signals:
            ticker = signal.get("ticker", "")

            if ticker in held_tickers:
                rejected.append({"ticker": ticker, "reason": "Already holding position"})
                continue

            sharpe = signal.get("sharpe_ratio", 0)
            if sharpe < RISK_CONFIG["min_sharpe_ratio"] and sharpe != 0:
                rejected.append({
                    "ticker": ticker,
                    "reason": f"Sharpe {sharpe:.2f} < {RISK_CONFIG['min_sharpe_ratio']}",
                })
                continue

            valid_signals.append(signal)

        if not valid_signals:
            return {
                "approved_trades": [],
                "rejected_trades": rejected,
                "halt": False,
                "hedge_mode": False,
                "hedge_actions": [],
                "risk_summary": (
                    f"No valid trades. Rejected {len(rejected)}. "
                    f"VIX: {vix_level}. Daily P&L: {daily_loss_pct:+.2%}. "
                    f"Positions: {len(positions)}."
                ),
            }

        # POSITION MANAGEMENT
        # Cap total positions (existing + new) to avoid over-buying
        MAX_TOTAL_POSITIONS = 5
        existing_count = len(positions)
        slots_available = max(0, MAX_TOTAL_POSITIONS - existing_count)

        if slots_available == 0:
            for s in valid_signals:
                rejected.append({"ticker": s.get("ticker", ""), "reason": f"Max {MAX_TOTAL_POSITIONS} positions reached"})
            return {
                "approved_trades": [],
                "rejected_trades": rejected,
                "halt": False,
                "hedge_mode": False,
                "hedge_actions": [],
                "risk_summary": (
                    f"Max {MAX_TOTAL_POSITIONS} positions reached ({existing_count} held). "
                    f"VIX: {vix_level}. Daily P&L: {daily_loss_pct:+.2%}."
                ),
            }

        # ── BACKTEST-MATCHING POSITION SIZING ──────────────────────────────────
        # This mirrors the exact formula in run_backtest_v2.py:
        #
        #   stop_dist      = stop_r (1.5) * ATR_5m
        #   stop_dist_pct  = stop_dist / entry_price
        #   leverage       = min(risk_pct / stop_dist_pct, 10.0)
        #   position_$     = equity * leverage
        #   qty            = position_$ / price
        #
        # PnL impact when stop hits = equity * risk_pct (5%) → exactly 1 risk-unit.
        # Buying power needed       = equity * leverage (up to 10x).
        # ─────────────────────────────────────────────────────────────────────────

        valid_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        target_risk_pct = RISK_CONFIG.get("risk_per_trade_pct", 0.05)   # 5%
        stop_r          = RISK_CONFIG.get("stop_r_atr", 1.5)             # 1.5 × ATR stop
        max_leverage    = RISK_CONFIG.get("max_leverage", 2.0)           # HARD CAP: 2x equity per trade

        # Track how much buying power we will commit this cycle
        # Alpaca Day Trading Buying Power is 4x Equity. We leave a 10% buffer.
        running_bp_used = 0.0
        approved        = []

        for signal in valid_signals:
            ticker        = signal.get("ticker", "")
            direction     = signal.get("direction", "long")
            confidence    = signal.get("confidence", 0)
            current_price = signal.get("current_price", 0)
            atr           = signal.get("atr_5m", 0)
            stop_loss     = signal.get("stop_loss", 0)

            if current_price <= 0:
                rejected.append({"ticker": ticker, "reason": "Zero price"})
                continue

            # 1. Compute stop distance ($ per share)
            if stop_loss > 0 and current_price > stop_loss:
                stop_dist_dollars = current_price - stop_loss
            elif atr > 0:
                stop_dist_dollars = stop_r * atr          # backtest default: 1.5 × ATR
            else:
                stop_dist_dollars = current_price * 0.015  # fallback 1.5%

            stop_dist_pct = stop_dist_dollars / current_price

            # 2. Leverage = backtest formula, capped at max_leverage
            leverage = min(target_risk_pct / stop_dist_pct, max_leverage) if stop_dist_pct > 0 else 1.0

            # 3. Position size in dollars and shares
            position_dollars = equity * leverage

            # 4. Safety: never exceed available buying power (leave 10% buffer)
            # Maximum allowed total investment limit is 4.0x equity (Alpaca DTBP)
            max_bp_this_trade = buying_power * 0.90 - running_bp_used
            if position_dollars > max_bp_this_trade:
                position_dollars = max_bp_this_trade

            qty = max(1, int(position_dollars / current_price))
            actual_cost = qty * current_price

            if actual_cost < 50:
                rejected.append({"ticker": ticker, "reason": "Calculated position too small"})
                continue

            if running_bp_used + actual_cost > buying_power * 0.90:
                rejected.append({"ticker": ticker, "reason": f"Insufficient buying power (${buying_power:,.0f})"})
                continue

            running_bp_used += actual_cost

            approved_trade = {
                "ticker":         ticker,
                "direction":      direction,
                "qty":            qty,
                "position_dollars": round(actual_cost, 2),
                "position_pct":   round((actual_cost / equity) * 100, 2) if equity > 0 else 0,
                "leverage":       round(leverage, 2),
                "kelly_fraction": 0,
                "sharpe_ratio":   signal.get("sharpe_ratio", 0),
                "confidence":     confidence,
                "stop_loss_pct":  RISK_CONFIG["default_stop_loss_pct"],
                "current_price":  current_price,
            }
            approved.append(approved_trade)

            logger.info(
                f"APPROVED: {direction.upper()} {qty} {ticker} "
                f"@ ${current_price:.2f} | stop_dist=${stop_dist_dollars:.3f} "
                f"({stop_dist_pct:.2%}) | lev={leverage:.2f}x | "
                f"deploy=${actual_cost:,.0f} ({(actual_cost/equity)*100:.1f}% eq)"
            )


        return {
            "approved_trades": approved,
            "rejected_trades": rejected,
            "halt": False,
            "hedge_mode": False,
            "hedge_actions": [],
            "risk_summary": (
                f"Approved {len(approved)}/{len(trade_signals)} trades (equal-weight). "
                f"Rejected {len(rejected)}. "
                f"VIX: {vix_level}. Daily P&L: {daily_loss_pct:+.2%}. "
                f"Positions: {len(positions)}."
            ),
        }

    def _calculate_daily_loss(self, positions: list[dict], equity: float) -> float:
        if not positions or equity <= 0:
            return 0.0
        total_unrealized_pl = sum(p.get("unrealized_pl", 0) for p in positions)
        return abs(min(0, total_unrealized_pl)) / equity

    def _generate_hedge_actions(self, positions: list[dict], equity: float) -> list[dict]:
        actions = []
        safe_haven_tickers = {"GLD", "TLT", "SLV", "UUP", "BND"}
        for pos in positions:
            ticker = pos.get("ticker", "")
            if ticker not in safe_haven_tickers:
                actions.append({"action": "close", "ticker": ticker,
                                "reason": "VIX hedge mode"})

        hedge_alloc = equity * 0.15
        actions.append({"action": "buy", "ticker": "GLD",
                        "dollars": round(hedge_alloc * 0.6, 2), "reason": "Hedge: gold"})
        actions.append({"action": "buy", "ticker": "TLT",
                        "dollars": round(hedge_alloc * 0.4, 2), "reason": "Hedge: treasuries"})
        return actions

    def _halt_response(self, reason: str) -> dict:
        return {
            "approved_trades": [], "rejected_trades": [],
            "halt": True, "hedge_mode": False, "hedge_actions": [],
            "risk_summary": f"HALT: {reason}",
        }

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.halted = False
        self.hedge_mode = False
        logger.info("Daily risk counters reset")
