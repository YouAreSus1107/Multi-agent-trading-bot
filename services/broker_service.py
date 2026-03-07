"""
War-Room Bot — Broker Service
Uses Alpaca API for paper/live trade execution and portfolio management.
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils.logger import get_logger, log_trade_decision

logger = get_logger("broker_service")


class BrokerService:
    """
    Alpaca-based broker service for trade execution.
    Defaults to paper trading for safety.
    """

    def __init__(self):
        is_paper = "paper" in ALPACA_BASE_URL.lower()
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=is_paper,
        )
        self.is_paper = is_paper
        logger.info(
            f"Broker initialized (mode: {'PAPER' if is_paper else 'WARNING: LIVE'})"
        )

    def get_account(self) -> dict:
        """
        Get current account information.

        Returns:
            {
                "equity": float,
                "buying_power": float,
                "cash": float,
                "portfolio_value": float,
                "day_trade_count": int,
                "pattern_day_trader": bool,
            }
        """
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            logger.error(f"Account fetch failed: {e}")
            return {}

    def get_positions(self) -> list[dict]:
        """
        Get all current portfolio positions.

        Returns:
            [
                {
                    "ticker": str,
                    "qty": float,
                    "side": str,
                    "entry_price": float,
                    "current_price": float,
                    "market_value": float,
                    "unrealized_pl": float,
                    "unrealized_pl_pct": float,
                }
            ]
        """
        try:
            positions = self.client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    "ticker": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    "entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_pl_pct": float(pos.unrealized_plpc) * 100,
                })
            return result
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return []

    def execute_trade(
        self,
        ticker: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: float = None,
    ) -> dict:
        """
        Execute a trade order.

        Args:
            ticker: Stock symbol (e.g., "XLE").
            side: "buy" or "sell".
            qty: Number of shares.
            order_type: "market" or "limit".
            limit_price: Required if order_type is "limit".

        Returns:
            {
                "order_id": str,
                "status": str,
                "ticker": str,
                "side": str,
                "qty": int,
                "filled_price": float or None,
            }
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            if order_type == "limit" and limit_price:
                order_data = LimitOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )

            order = self.client.submit_order(order_data)

            result = {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }

            log_trade_decision(
                logger, "broker", ticker, side,
                f"Order submitted ({order_type})",
                {"qty": qty, "order_id": result["order_id"]},
            )

            return result

        except Exception as e:
            logger.error(f"Trade execution failed for {ticker}: {e}")
            return {
                "order_id": None,
                "status": "failed",
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "error": str(e),
            }

    def close_position(self, ticker: str) -> dict:
        """Close an entire position for a ticker."""
        try:
            self.client.close_position(ticker)
            logger.info(f"Closed position: {ticker}")
            return {"status": "closed", "ticker": ticker}
        except Exception as e:
            logger.error(f"Close position failed for {ticker}: {e}")
            return {"status": "failed", "ticker": ticker, "error": str(e)}

    def close_all_positions(self) -> dict:
        """Emergency: close ALL positions (kill switch)."""
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.warning("EMERGENCY: Closed ALL positions")
            return {"status": "all_closed"}
        except Exception as e:
            logger.error(f"Close all positions failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_open_orders(self) -> list[dict]:
        """Get all open (pending) orders."""
        try:
            orders = self.client.get_orders()
            return [
                {
                    "order_id": str(o.id),
                    "ticker": o.symbol,
                    "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                    "qty": float(o.qty),
                    "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                    "order_type": o.type.value if hasattr(o.type, 'value') else str(o.type),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Open orders fetch failed: {e}")
            return []

    def cancel_all_orders(self) -> dict:
        """Cancel all open/pending orders before EOD close."""
        try:
            self.client.cancel_orders()
            logger.info("Cancelled all pending orders")
            return {"status": "cancelled"}
        except Exception as e:
            logger.warning(f"Cancel all orders failed (may have no open orders): {e}")
            return {"status": "ok", "note": str(e)}

    def close_day_trade_positions(self, day_trade_tickers: set) -> list[dict]:
        """
        EOD smart close: liquidate only day-trade positions, not mega-cap holds.
        Returns list of closed position results with realized P&L.
        """
        positions = self.get_positions()
        results = []

        for pos in positions:
            ticker = pos.get("ticker", "")
            if ticker not in day_trade_tickers:
                logger.info(f"EOD: Keeping mega-cap hold {ticker}")
                continue

            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            pl_pct = pos.get("unrealized_pl_pct", 0)
            pl_dollars = pos.get("unrealized_pl", 0)

            try:
                self.client.close_position(ticker)
                logger.info(f"EOD: Closed day-trade {ticker} | P&L: {pl_pct:+.2f}% (${pl_dollars:+.0f})")
                results.append({
                    "ticker": ticker,
                    "status": "closed",
                    "entry_price": entry,
                    "exit_price": current,
                    "realized_pl_pct": pl_pct,
                    "realized_pl_dollars": pl_dollars,
                    "qty": pos.get("qty", 0),
                })
            except Exception as e:
                logger.error(f"EOD close failed for {ticker}: {e}")
                results.append({
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(e),
                    "realized_pl_pct": pl_pct,
                })

        return results

    def get_todays_filled_orders(self) -> list[dict]:
        """Fetch all filled (closed) orders from the current trading session timeframe."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
            orders = self.client.get_orders(req)
            results = []
            for o in orders:
                if getattr(o, "filled_avg_price", None):
                    results.append({
                        "ticker": o.symbol,
                        "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                        "filled_price": float(o.filled_avg_price),
                        "filled_at": o.filled_at.isoformat() if getattr(o, "filled_at", None) else None,
                        "qty": float(o.filled_qty) if getattr(o, "filled_qty", None) else float(o.qty),
                    })
            return results
        except Exception as e:
            logger.error(f"Failed to fetch filled orders: {e}")
            return []
