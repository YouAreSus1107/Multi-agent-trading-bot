# Changelog

## 2026-04-02 — Strategy Rebuild & Bug Fix Sprint

### Strategy Rebuild

**Entry logic rewrite** (`backtest/run_backtest_v2.py`)

- Renamed `evaluate_bull_entry()` → `evaluate_momentum_entry()`
- Renamed `evaluate_dip_buy_entry()` → `evaluate_mean_reversion_entry()`
- Momentum now correctly requires VWAP Z-score > threshold (price *above* VWAP). Previous logic was inverted — was buying pullbacks, not breakouts.
- Mean reversion now requires volume spike (`volume_ratio >= rev_vol_spike_min`) AND `smart_bounce` confirmation. Previously lacked the key capitulation signal.
- `smart_bounce` moved from momentum path (incorrect) to mean reversion path (correct).

**Regime model replaced** (`backtest/run_backtest_v2.py`)

- Replaced slow HMM-based regime with intraday `_intraday_regime()` using SPY 5m EMA(20).
- Regime no longer hard-switches between strategies. Instead scales position size: bull=1.0×, chop=0.6×, bear=0.4×.
- Eliminates the fundamental problem of two long-only strategies providing no downside protection.

**Parameter reduction: 17 → 8** (`backtest/optimize_v2.py`)

- Optuna search space cut from 17 to 8 parameters.
- New params: `mom_vwap_z_min`, `mom_vol_ratio_min`, `mom_delta_min`, `rev_vwap_z_max`, `rev_vol_spike_min`, `stop_r`, `target_r`, `risk_per_trade`.
- Old params removed: `long_hybrid`, `long_exec`, `long_vwap`, `dip_hybrid`, `dip_exec`, `dip_vwap`, `long_bounce_pct`, `dip_bounce_pct`, and others.
- Warm-start seeds updated to match new space.

**Mean reversion exit logic**

- VWAP reversion exit fires when Z-score >= -0.5 (price mean-reverts to VWAP).
- 30-bar time stop replaces old dip-buy timeout.

**`test_params.txt` corrected**

- File had stale pre-rebuild column names (`long_hybrid`, `dip_exec`, etc.).
- Overwritten with correct new-format baseline: `mom_vwap_z_min=0.3`, `mom_vol_ratio_min=1.3`, `mom_delta_min=0.58`, `rev_vwap_z_max=-2.5`, `rev_vol_spike_min=2.0`, `stop_r=1.5`, `target_r=2.0`, `risk_per_trade=0.05`.

---

### Bug Fixes

**`utils/quant_engine.py`** — Added `volume_ratio = volume / vol_sma_20` column. Required by new mean reversion entry gate; available in both live and backtest paths.

**`services/discord_notifier.py`** — Updated stale parameter references (`long_hybrid`, `dip_hybrid`, etc.) to new parameter names.

---

### Prior Bug Fixes (this sprint)

| Commit | Description |
|---|---|
| `13a9dc3` | Wire `risk_per_trade` into Optuna search space; consolidate param files |
| `e41fe0b` | Dynamic Volatility Parity position sizing with conviction scaling and concentration caps |
| `d16575f` | Bear regime reshaped to buy-the-dip mean reversion; refine `auto_adjust` alignment |
| `4d57992` | Fix trailing PnL double-compound bug; fix yfinance `auto_adjust` stock-split scaling |
| `4263196` | Fix inverse ETF boundary constraint causing zero-trade optimizer trials; add total return tracking |
