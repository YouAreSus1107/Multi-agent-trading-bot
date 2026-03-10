"""
War-Room Bot — Structured Logger
JSON-formatted logging for audit trail of every decision, trade, and kill-switch activation.
"""

import logging
import json
import sys
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging and audit trails."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", "system"),
            "action": getattr(record, "action", ""),
            "message": record.getMessage(),
        }

        # Attach extra data if present
        if hasattr(record, "data"):
            log_entry["data"] = record.data

        return json.dumps(log_entry, default=str)


def get_logger(agent_name: str = "system") -> logging.Logger:
    """
    Creates a logger instance tagged with the agent name.

    Usage:
        logger = get_logger("sigint")
        logger.info("Scanning news", extra={"agent": "sigint", "action": "scan", "data": {...}})
    """
    logger = logging.getLogger(f"warroom.{agent_name}")

    if not logger.handlers:
        if os.environ.get("DISABLE_FILE_LOGGING") == "1":
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.WARNING)
        else:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "warroom.log")
            
            from logging.handlers import RotatingFileHandler
            
            class SafeRotatingFileHandler(RotatingFileHandler):
                """RotatingFileHandler that silently handles PermissionError on Windows/OneDrive."""
                def doRollover(self):
                    try:
                        super().doRollover()
                    except PermissionError:
                        pass  # OneDrive or another process has the backup locked — skip rotation
            
            # Cap the log file at 5MB with 1 backup, automatically deleting old bloat
            handler = SafeRotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=1, encoding='utf-8')
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        logger.propagate = False

    return logger


def log_trade_decision(logger: logging.Logger, agent: str, ticker: str,
                       side: str, reason: str, data: dict = None):
    """Convenience function for logging trade decisions."""
    logger.info(
        f"TRADE DECISION: {side} {ticker} — {reason}",
        extra={
            "agent": agent,
            "action": "trade_decision",
            "data": {
                "ticker": ticker,
                "side": side,
                "reason": reason,
                **(data or {}),
            },
        },
    )


def log_kill_switch(logger: logging.Logger, trigger: str, vix_level: float = None,
                    daily_loss_pct: float = None):
    """Convenience function for logging kill-switch activations."""
    logger.warning(
        f"KILL SWITCH ACTIVATED: {trigger}",
        extra={
            "agent": "risk_manager",
            "action": "kill_switch",
            "data": {
                "trigger": trigger,
                "vix_level": vix_level,
                "daily_loss_pct": daily_loss_pct,
            },
        },
    )
