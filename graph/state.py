"""
War-Room Bot -- LangGraph Shared State (Day Trading Architecture)
"""

from typing import TypedDict


class WarRoomState(TypedDict):
    """Shared state flowing through the trading pipeline."""

    # === Analyst Outputs ===
    technical_data: dict          # {ticker: {trend_score, rsi, signals, ...}}
    fundamentals_data: dict       # {fundamentals: [...], summary: "..."}
    news_data: dict               # {events, niche_tickers, sector_impact, market_mood}
    sentiment_data: dict          # {ticker_sentiments, market_mood, fear_greed}
    quant_data: dict              # {quant_data: {ticker: {gbm, factors, xgboost}}}

    # === Research Team Output ===
    research_decisions: list      # [{ticker, action, conviction, entry, target, stop}]
    market_stance: str            # aggressive_bull / cautious / defensive
    overall_strategy: str         # Current period strategy text
    research_notes: str

    # === Trader Agent Output ===
    trader_signals: list          # Signals ready to be executed right now
    trader_strategy: str          # Current execution logic narrative

    # === Risk Agent Output ===
    approved_trades: list
    rejected_by_risk: list
    halt: bool
    hedge_mode: bool
    hedge_actions: list
    risk_summary: str

    # === Execution Output ===
    executed_trades: list
    execution_errors: list
    exit_signals: list            # [{ticker, reason, urgency}]

    # === Portfolio ===
    portfolio: dict
    positions: list
    vix_level: float

    # === Metadata ===
    cycle_number: int
    cycle_timestamp: str
    candidate_tickers: list       # All tickers under consideration this cycle
    news_items: list              # Raw news items
