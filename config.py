"""
War-Room Bot -- Central Configuration
Day-trading focused: high-beta stocks, fast analysis, tight risk controls.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================
# API Keys
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "war-room-events")
PUTER_API_TOKEN = os.getenv("PUTER_API_TOKEN", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ============================================
# LLM Configuration
# Failover: Gemma 3 27B -> Groq -> Puter -> OpenAI
# ============================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL_GEMMA = "gemini-2.0-flash-lite"    # Free on Google AI Studio, fast + cheap
LLM_MODEL_GROQ = "llama-3.3-70b-versatile"
LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_TEMPERATURE = 0.15

# ============================================
# Day-Trading Tickers (high beta, high leverage)
# These are volatile individual stocks good for intraday moves
# ============================================
DAY_TRADE_TICKERS = {
    "momentum": {
        "tickers": ["PLTR", "MSTR", "SMCI", "IONQ", "RKLB", "MARA"],
        "description": "High-momentum tech/AI/crypto names with big intraday swings",
    },
    "fintech": {
        "tickers": ["COIN", "HOOD", "SOFI", "XYZ"],
        "description": "Fintech -- sentiment-driven, volatile",
    },
    "ev_energy": {
        "tickers": ["RIVN", "LCID", "FSLR", "ENPH"],
        "description": "EV and clean energy -- policy and sentiment driven",
    },
    "defense_geo": {
        "tickers": ["LMT", "RTX", "NOC", "GD", "HII"],
        "description": "Defense -- geopolitical catalyst driven",
    },
    "mega_cap": {
        "tickers": ["NVDA", "AMD", "TSLA", "META", "GOOGL"],
        "description": "Mega-caps with high daily volume and options flow",
    },
    "speculative": {
        "tickers": ["SNAP", "DKNG", "AFRM", "UPST"],
        "description": "High-risk speculative plays, large % moves",
    },
    "small_midcap_catalyst": {
        "tickers": ["ACHR", "JOBY", "SOUN", "RXRX", "IREN", "CLSK", "MVIS", "PRCT", "APLD", "WOLF", "CLOV", "HIMS"],
        "description": "Small/mid-cap high-beta names with catalysts — earnings, FDA, contracts, momentum",
    },
}

# Portfolio allocation split between mega-cap holds and active day-trade bucket
PORTFOLIO_ALLOCATION = {
    "mega_cap_pct": 0.60,      # 60% BP for mega-cap / established holds (TSLA, META, NVDA)
    "day_trade_pct": 0.40,     # 40% BP for small/mid-cap catalyst day trades
    "mega_cap_tickers": ["NVDA", "AMD", "TSLA", "META", "GOOGL", "AAPL", "MSFT", "AMZN"],
}

# Flatten all day-trade tickers (used as the seed for dynamic ticker tracking)
ALL_DAY_TRADE_TICKERS = []
for group in DAY_TRADE_TICKERS.values():
    ALL_DAY_TRADE_TICKERS.extend(group["tickers"])
ALL_DAY_TRADE_TICKERS = list(set(ALL_DAY_TRADE_TICKERS))

# Legacy sector mapping for news analysis
SECTOR_TICKERS = {
    "energy": ["XOM", "CVX", "OXY", "COP", "SLB"],
    "defense": ["LMT", "RTX", "NOC", "GD", "HII"],
    "tech": ["NVDA", "AMD", "PLTR", "SMCI", "IONQ", "RKLB"],
    "crypto": ["MSTR", "MARA", "COIN"],
    "fintech": ["HOOD", "SOFI", "SQ", "AFRM", "UPST"],
    "ev": ["RIVN", "LCID", "TSLA"],
    "social": ["SNAP", "META"],
    "healthcare": ["UNH", "LLY", "PFE"],
}

# Market reference tickers
REFERENCE_TICKERS = {
    "spy": "SPY",
    "qqq": "QQQ",
    "vix_proxy": "VIXY",
    "gold": "GLD",
}

# ============================================
# News Keywords
# ============================================
WAR_KEYWORDS = [
    "Iran", "Strait of Hormuz", "Houthi", "enrichment",
    "IRGC", "sanctions", "Persian Gulf",
    "drone strike", "missile attack", "oil tanker",
    "nuclear", "JCPOA", "proxy war",
    "CENTCOM", "Hezbollah", "Red Sea shipping",
    "Pentagon", "defense spending", "arms deal",
    "cyber attack", "retaliatory strike",
]

MARKET_KEYWORDS = [
    "stock market", "S&P 500", "Federal Reserve", "interest rates",
    "inflation", "GDP", "earnings report", "trade war",
    "tariffs", "recession", "tech stocks", "AI stocks",
    "cryptocurrency", "bitcoin", "oil prices",
    "supply chain", "semiconductor", "OPEC",
    "Palantir", "defense contract", "government contract",
    "meme stock", "short squeeze", "options flow",
]

CATALYST_KEYWORDS = [
    "earnings beat", "earnings surprise", "FDA approval", "FDA breakthrough",
    "contract win", "government contract awarded", "short squeeze",
    "unusual options activity", "small cap momentum", "mid cap breakout",
    "analyst upgrade", "price target raised", "buyout", "merger",
    "clinical trial success", "product launch", "partnership announced",
]

# ============================================
# Risk Management -- Day Trading
# Tighter stops, faster profits than swing trading
# ============================================
RISK_CONFIG = {
    "vix_kill_switch": 45,              # Higher threshold for day trading
    "max_position_pct": 0.20,           # Up to 20% per trade
    "max_daily_loss_pct": 0.05,         # 5% daily loss halt
    "min_sharpe_ratio": -2.0,            # Very low -- let Research Team debate decide quality
    "risk_free_rate": 0.05,
    "default_stop_loss_pct": 0.015,     # 1.5% stop loss (tight for day trades)
    "take_profit_pct": 0.03,            # 3% take profit
    "max_trades_per_cycle": 5,          # Cap per fast cycle
}

# ============================================
# Strategy Thresholds
# ============================================
STRATEGY_CONFIG = {
    "min_sentiment_score": 0.40,
    "min_trend_score": 45,              # Lower bar to enter
    "rsi_overbought": 80,
    "rsi_oversold": 25,
    "bollinger_period": 20,
    "bollinger_std_dev": 2,
    "rsi_period": 14,
}

# ============================================
# Timing Configuration
# ============================================
FAST_LOOP_SECONDS = 120                 # Technical + Fundamentals (2 min)
SLOW_LOOP_SECONDS = 300                 # News + Sentiment (5 min)
LOOP_INTERVAL_SECONDS = FAST_LOOP_SECONDS

# US Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ============================================
# Pinecone / Vector DB Configuration
# ============================================
VECTOR_EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536
VECTOR_TOP_K = 5
