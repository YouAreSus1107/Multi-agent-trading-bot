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

        # Only approve up to `slots_available` trades (sorted by confidence)
        valid_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        valid_signals = valid_signals[:slots_available]

        # EQUAL-WEIGHT ALLOCATION based on REMAINING capacity
        # Calculate how much capital is already deployed
        deployed_value = sum(
            abs(p.get("market_value", p.get("qty", 0) * p.get("current_price", 0)))
            for p in positions
        )
        remaining_equity = max(0, equity - deployed_value)

        # Risk budget per trade: target risking 1% of total equity per trade
        # Equation: Shares = Risk_Dollars / Stop_Distance_Dollars
        target_risk_pct = RISK_CONFIG.get("risk_per_trade_pct", 0.01)
        risk_budget_dollars = equity * target_risk_pct

        # Keep 25% cash reserve minimum
        allocatable = min(remaining_equity * 0.75, buying_power * 0.85)

        if allocatable < 200:
            for s in valid_signals:
                rejected.append({"ticker": s.get("ticker", ""), "reason": f"Low buying power (${buying_power:,.0f})"})
            return {
                "approved_trades": [],
                "rejected_trades": rejected,
                "halt": False,
                "hedge_mode": False,
                "hedge_actions": [],
                "risk_summary": (
                    f"Insufficient allocatable capital (${allocatable:,.0f}). "
                    f"BP: ${buying_power:,.0f}. Deployed: ${deployed_value:,.0f}. "
                    f"Positions: {existing_count}."
                ),
            }

        max_per_trade_equity = equity * RISK_CONFIG["max_position_pct"]

        approved = []
        running_cost = 0  # Track cumulative cost to avoid over-spending

        for signal in valid_signals:
            ticker = signal.get("ticker", "")
            direction = signal.get("direction", "long")
            confidence = signal.get("confidence", 0)
            current_price = signal.get("current_price", 0)
            
            # Use 1.5 ATR 5m as the default stop distance, or fallback to 2%
            atr = signal.get("atr_5m", 0)
            if atr > 0:
                stop_distance = 1.5 * atr
            elif current_price > 0:
                 stop_distance = current_price * 0.02
            else:
                 stop_distance = 1.0

            if current_price and current_price > 0:
                # Volatility parity sizing: Shares = Risk $ / Stop Distance $
                ideal_qty = int(risk_budget_dollars / stop_distance)
                ideal_cost = ideal_qty * current_price
                
                # Cap the trade cost to the max_per_trade_equity and allocatable limits
                capped_cost = min(ideal_cost, max_per_trade_equity, allocatable - running_cost)
                qty = max(1, int(capped_cost / current_price))
                actual_cost = qty * current_price
            else:
                qty = max(1, int(risk_budget_dollars / stop_distance / 100)) # dummy calculation
                actual_cost = risk_budget_dollars # dummy 

            if running_cost + actual_cost > allocatable or actual_cost < 50:
                rejected.append({"ticker": ticker, "reason": "Insufficient remaining allocation"})
                continue

            running_cost += actual_cost

            approved_trade = {
                "ticker": ticker,
                "direction": direction,
                "qty": qty,
                "position_dollars": round(actual_cost, 2),
                "position_pct": round((actual_cost / equity) * 100, 2) if equity > 0 else 0,
                "kelly_fraction": 0,
                "sharpe_ratio": signal.get("sharpe_ratio", 0),
                "confidence": confidence,
                "stop_loss_pct": RISK_CONFIG["default_stop_loss_pct"],
                "current_price": current_price,
            }
            approved.append(approved_trade)

            logger.info(
                f"APPROVED: {direction.upper()} {qty} {ticker} "
                f"(${actual_cost:,.0f}, {actual_cost/equity:.1%} of portfolio, "
                f"remaining alloc: ${allocatable - running_cost:,.0f})"
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
