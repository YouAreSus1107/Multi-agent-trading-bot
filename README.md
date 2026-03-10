# War-Room AI Trading Bot

A high-frequency AI-driven day trading architecture optimized for high-beta stocks, featuring a dual-speed pipeline (2 min / 5 min), a multi-agent Bull/Bear debate structure, and quantitative execution triggers based on institutional microstructure models.

## 🚀 Key Features

*   **Dual-Speed Pipeline**: Fast technical and quantitative loops (every 2 minutes) coupled with slower, deeper fundamental, news, and sentiment loops (every 5-6 minutes to conserve tokens).
*   **V2 3-Layer Execution Architecture**: 
    1. **Rank**: Daily universe scoring of S&P 500 components (T-1 data).
    2. **Regime**: HMA-Kalman filter on SPY to dictate Bull vs Bear posture.
    3. **Intraday Execution**: High-conviction entry gating with volatility-parity risk sizing and time-locked stops.
*   **Multi-Agent Research Team**: Utilizes specialized LLMs (Gemma, Llama, GPT) to argue Long vs Short on tickers based on real-time data. A Moderator agent weighs the Bull conviction against the Bear risk assessment.
*   **Institutional Quant Engine**: Employs VWAP Z-scores (mean reversion), Volume Delta (smart money flow/bounces), and ATR limits for risk.
*   **Dynamic Trailing Stops (v2.1)**: Trailing stops are time-locked at market open (-4% circuit breaker) to survive morning volatility, then ratcheted upward driven by 3.0x ATR on 5-minute charts. Stops never retreat.
*   **Automated EOD Liquidation**: Flushes all pure day-trade/catalyst-tier positions automatically at 3:57 PM ET, leaving the 60% Mega-Cap (TSLA/NVDA/META) core intact. Generates an End-of-Day assessment report grading strategy and timing.
*   **Overnight Phase**: Wakes up iteratively every hour while the market is closed to evaluate overnight geopolitical news and prep tomorrow's watch lists for the pre-market briefing. 

## 🧠 LangGraph Workflow Architecture

The execution runs in an acyclic graph structure every iteration:

1.  **Ingest Node**: Gathers live portfolio buying-power, current positions, macro risk regime (VIX spikes), and establishes the active candidate ticker sweep.
2.  **Slow Analysts (Every ~6 min)**:
    *   **News Analyst**: Ingests breaking catalyst news (FDA approvals, government contracts, earnings gaps).
    *   **Sentiment Analyst**: Cross-references the candidate ticker lists against news to generate a polarized score matrix.
3.  **Fast Analysts (Every 2 min)**:
    *   **Technical Analyst**: Generates standard indicator arrays (support/resistance, EMA/MACD momentum scores).
    *   **Quant Analyst**: Aggregates mathematical matrices: standard deviation away from intraday VWAP, Buy vs Sell microstructure volume ratio, smart-bounce setups, and statistical XGBoost probabilities.
    *   **Fundamentals Analyst**: Reviews financials and ratios.
4.  **Research Debate**: The Bull and Bear LLMs argue the trade viability. If consensus isn't reached, a Moderator agent renders a final `buy`, `sell`, or `avoid` verdict containing a mathematically driven conviction score.
5.  **Trader Agent**: Holds "buy" signals approved by the Research Team in a *pending queue*. The trade only hits the wire when the **Quant** threshold is met perfectly (e.g., waiting for the exact minute the price wicks off VWAP Support). Once held, manages ATR exits natively via memory lookup.
6.  **Risk Manager**: Dynamically sizes the quantities to execute based on active VIX constraints, daily max-loss ceilings, and remaining liquid cash allocations partitioned by strategy tier (Mega-Cap hold vs Catalyst swing).
7.  **Execution**: Wraps the finalized arrays to push straight line-level orders to Alpaca. Stores all trading logic and decision pathways back into long-term recursive memory (`execution_memory.json` / Pinecone context index).

## 📦 Logistics & Tech Stack

This framework operates heavily on LLM-as-a-judge patterns mapped across diverse endpoints. 

*   **Core**: Python 3.10+, `langgraph` framework
*   **Data Feeds**: Finnhub (Fundamentals/News), Tavily (Search indexing), Yahoo Finance / Web Scraping logic via internal suites.
*   **Brokerage**: Alpaca (Paper/Live via API) + Market Data provider
*   **LLM Failover Array**: Google Gemini 2.0 Flash → Groq Llama 3 → OpenAI GPT-4o-mini → Puter  
    *(Automatically falls back to limit API rate-limit bottlenecks during peak 9:30 AM volatility surges)*.
*   **Memory Integration**: Chroma/Pinecone vector storage for event pattern matching (e.g., querying "How did LMT behave last time there was a drone strike in the Red Sea?").

### 🧪 V2 Smart Parameter Optimizer

The repository includes a Bayesian Optimization suite using Optuna TPE to find robust cross-regime trading parameters before pushing to live execution.
```bash
# Run the optimizer overnight across multi-month evaluation windows
python backtest/optimize_v2.py

# Export the best discovered parameters to test_params_optimized.txt
python backtest/optimize_v2.py --export-best
```

### Quick Setup

1.  Clone the repository and install essential requirements:
    ```bash
    pip install -r requirements.txt
    ```
2.  Populate `.env` by duplicating `.env.example`. Supply all specialized API/Broker keys. 
3.  Execute the script loop manually, or via a bound `.bat` startup shortcut. 
    ```bash
    python main.py
    ```

**Startup Arguments:**
*   `--force`: Forego US Market hours constraints; force the graph structure to execute trade cycles immediately.
*   `--once`: Run exactly one complete iteration of the pipeline (used predominantly for regression testing).
*   `--assess`: Immediately generate the AI-driven End-Of-Day assessment grading review without liquidating current accounts.
