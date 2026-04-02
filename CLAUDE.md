# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

War-Room is an AI-driven intraday trading bot for high-beta stocks. It runs a 3-layer pipeline:
1. **Layer 1**: Daily S&P 500 universe ranking (alpha/beta/volatility/momentum) using T-1 data
2. **Layer 2**: Regime detection (Bull/Bear/Choppy) — HMM-based in backtest, reactive SPY/VIX in live
3. **Layer 3**: Intraday 5-minute execution loop — analysts → debate → risk → execute

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot (standard mode — auto-detects market hours)
python main.py

# Bypass market hours (for testing)
python main.py --force

# Single cycle then exit (regression check)
python main.py --once

# Generate EOD assessment report
python main.py --assess

# Run unit tests
pytest tests/
pytest tests/test_services.py   # indicator math
pytest tests/test_risk.py       # risk manager logic
pytest tests/test_macro.py      # regime detection
pytest tests/test_strategy.py   # strategy logic

# Backtesting
python backtest/run_backtest_v2.py
python backtest/optimize_v2.py
python backtest/optimize_v2.py --resume          # resume from SQLite study
python backtest/optimize_v2.py --show-best 20    # top 20 param sets
python backtest/optimize_v2.py --export-best     # export to test_params.txt
python backtest/optimize_v2.py --n-jobs 2        # parallel workers
```

## Architecture

### Main Loop Lifecycle (`main.py`)

The bot has distinct operating phases, all timezone-aware (US Eastern):

| Phase | Time (ET) | Behavior |
|---|---|---|
| **Premarket briefing** | 9:10–9:30 | Reads yesterday's EOD report, extracts lessons/watch/avoid lists |
| **Active trading** | 9:30–15:57 | 5-min LangGraph cycles (ingest → analysts → debate → risk → execute) |
| **EOD liquidation** | 15:57 | Hard-closes all day-trade positions; keeps mega-cap holds overnight |
| **Overnight** | 16:00–9:10 | Hourly wake-ups to evaluate geopolitical news for next-day prep |

Each 5-min cycle creates a **fresh** `WarRoomState` dict — there is no persistent DAG state between cycles. Only `execution_memory.json` and `cycle_memory.json` persist knowledge across cycles.

### LangGraph Workflow (`graph/workflow.py`)

The DAG state machine (`WarRoomState` in `graph/state.py`) flows through these nodes:

```
ingest_data → slow_analysts → fast_analysts → research → trader → risk →[conditional]→ execute | END
```

- **ingest_data**: Fetches portfolio, VIX, regime; loads Layer 1 ranked universe (top 15); syncs execution_memory against broker positions
- **slow_analysts**: News + Sentiment — runs every 3 cycles, cached between (LLM-heavy, token-saving)
- **fast_analysts**: Technical (RSI/Bollinger/EMA/MACD), Quant (VWAP Z-score/Volume Delta/ATR), Fundamentals (LLM, also 3-cycle cached)
- **research**: Bull vs Bear LLMs debate each ticker; Moderator renders verdict + conviction score (0-100)
- **trader**: Entry gate (quant_entry AND exec_score ≥ 36); manages pending entry queue with 15-cycle (~75 min) expiry; can **auto-queue tickers** meeting quant criteria without research debate
- **risk**: VIX kill switch (>45 → hedge), daily max-loss halt (5%), volatility-parity position sizing
- **execute**: Submits orders to Alpaca; persists decision pathways; stores event patterns in vector DB

### Key Module Map

| Path | Purpose |
|---|---|
| `config.py` | All strategy/risk thresholds, ticker universes, API keys |
| `agents/research_team.py` | Bull/Bear LLM debaters + Moderator consensus (MIN_CONVICTION ≥ 75 to enter pending queue) |
| `agents/trader_agent.py` | Entry gate logic; pending entry queue; ATR trailing stop management |
| `agents/risk_agent.py` | VIX kill switch, daily loss halt, vol-parity position sizing |
| `agents/quant_agent.py` | Market scanner, Volume Delta, XGBoost/GBM signals |
| `services/market_service.py` | Finnhub + Alpaca data; reactive regime detection (SPY/GLD/VIX) |
| `services/broker_service.py` | Alpaca API — execute trades, manage positions |
| `services/universe_service.py` | Daily S&P 500 ranking (T-1), caches per day |
| `utils/quant_engine.py` | VWAP Z-scores, Volume Delta, ATR (Wilder's smoothing), smart-bounce detection |
| `utils/llm_factory.py` | LLM failover chain: Gemini 2.0 Flash Lite → Groq Llama 3.3 70B → GPT-4o-mini |
| `utils/memory.py` | Persistent `execution_memory.json` — pending entries, fills, ATR stops |
| `backtest/run_backtest_v2.py` | Full 3-layer backtest runner (~1086 lines) |
| `backtest/optimize_v2.py` | Bayesian (Optuna TPE) optimization of 16 parameters across 4 regimes |
| `backtest/data_loader_v2.py` | Parses Kaggle S&P 500 CSV; computes universe ranking; fetches 5m intraday bars |
| `backtest/regime_v3.py` | HMM-based regime model (Bull/Bear/Choppy) with hysteresis (min_dwell_days=5) |

### Position Sizing (Shared Between Live & Backtest)

Both use the same volatility-parity formula (`agents/risk_agent.py`):
```
stop_dist = stop_r × ATR_5m          (default stop_r = 1.5)
leverage  = min(risk_pct / (stop_dist / price), max_leverage)
position  = equity × leverage
qty       = int(position / price)
```

### Persistent State Files

| File | Purpose |
|---|---|
| `data/execution_memory.json` | Pending entries, ATR stops, trailing stops, entry prices (critical for live trading) |
| `data/cycle_memory.json` | Position theses, trade history, cycle summaries |
| `data/fundamental_profiles.json` | Daily T-1 universe ranking (cached per day) |
| `data/research_log.jsonl` | Append-only log of all Research Team debates |
| `data/reports/` | EOD markdown reports |
| `backtest/optimize_v2_study.db` | Optuna study (SQLite); resume with `--resume` |

### Backtest Optimization

`optimize_v2.py` tunes 16 parameters across 4 historical regimes: Jan 2021 bull, Aug 2022 bear, May 2023 AI bull, Oct 2021 choppy. Uses warm-start with 3 seed param sets (baseline, selective, permissive). Objective is a composite score: 0.15×PF + 0.15×WR + 0.25×avg_pnl + 0.30×total_return + 0.15×(1/drawdown). Hard penalties for >40% drawdown or PF < 0.8. Best params export to `backtest/test_params.txt` and `backtest/v2_optimization_results.csv`.

## Live vs Backtest Divergence Points

These are critical to understand when porting backtest improvements to live or vice versa:

| Aspect | Live | Backtest |
|---|---|---|
| **Regime model** | Reactive SPY/GLD/VIX signals (`market_service.py`). `universe_service.get_todays_regime()` **hardcodes "bull"** — regime switching not yet implemented in live | HMM-based multi-factor model (`regime_v3.py`) with trend scores, volatility features, chop index, hysteresis |
| **Entry logic** | Single gate: `quant_entry AND exec_score >= 36` — no regime-dependent thresholds | Dual-regime: separate `evaluate_momentum_entry()` and `evaluate_mean_reversion_entry()` with distinct VWAP/vol/delta thresholds |
| **Max leverage** | 2.0× (`risk_agent.py`) | 10.0× (config in `optimize_v2.py`) |
| **LLM agents** | Full Research Team debate + Moderator consensus | Pure quant — no LLM involvement |
| **Data source** | Alpaca API → yfinance fallback for quotes/candles | Kaggle S&P 500 CSV (daily) + yfinance (5m intraday) |
| **auto_adjust** | N/A (real-time prices) | `True` for SPY/normal tickers, `False` for inverse ETFs — critical for correct returns |

## Non-Obvious Gotchas

- **Warmup period**: Backtest skips first 25% of 5m bars per day (~80 min) to let indicators stabilize
- **Dual-class shares**: Data loader skips `BF.B`, `BRK.B` due to Alpaca API routing bugs
- **Cycle-based state**: Each 5-min cycle gets a fresh `WarRoomState` — only execution_memory.json preserves state across cycles
- **Quant auto-entry**: Trader agent can auto-queue tickers meeting quant criteria WITHOUT research debate (autonomous quant-only entry path)
- **No re-entry same bar**: Pending entries that expire cannot re-enter on the same bar (whipsaw prevention)
- **TSL engagement gate**: Trailing stop only activates after trade moves ≥ 1× ATR in profit — prevents premature tiny-loss exits on 5m bars

## Required Environment Variables (`.env`)

```
ALPACA_API_KEY, ALPACA_SECRET_KEY   # broker
FINNHUB_API_KEY                     # market data / news
TAVILY_API_KEY                      # web search
PINECONE_API_KEY                    # vector DB (war-event patterns)
OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY  # LLM providers
DISCORD_WEBHOOK_URL                 # trade alerts
```
