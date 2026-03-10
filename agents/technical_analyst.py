"""
Technical Analyst Agent — Hybrid Scoring System
Fast loop (every 2 min). Pure math, no LLM.

Composite Score (0–100):
  TREND BASELINE  (max 40 pts): RSI, MACD, EMA crossover, Bollinger
  EXECUTION TRIGGER (max 60 pts): ATR_14, VWAP Z-Score, Volume Delta

The Trend Baseline answers: "Is the macro setup favorable?"
The Execution Trigger answers: "Is THIS the right moment to pull the trigger?"
"""

from config import STRATEGY_CONFIG
from utils.indicators import (
    compute_trend_score, compute_daily_returns, compute_sharpe_ratio,
    compute_ema_trend_bias, compute_vwap, detect_rsi_exhaustion,
    detect_parabolic_reversal, detect_stop_hunt_zones,
    detect_mm_refill, detect_first_hour_trend,
)
from utils.quant_engine import (
    build_quant_metrics,
    precompute_quant_metrics_for_ticker,
    precompute_mtf_tsl_for_ticker,
)
from utils.logger import get_logger
from services.market_service import MarketService
from datetime import datetime, timezone

logger = get_logger("technical_analyst")

market_service = MarketService()


class TechnicalAnalyst:
    """
    Hybrid analyst: classic indicators (trend baseline) + quant metrics (execution trigger).
    Runs every 2 minutes. No LLM needed — pure math.
    """

    def __init__(self):
        self._daily_cache = {}

    def analyze(self, tickers: list[str], positions: list[dict] = None) -> dict:
        """
        Compute hybrid technical profiles for all tickers.
        Skips daily 60-day candle fetching for tickers already in positions.
        """
        profiles = {}
        if positions is None:
            positions = []
        held_tickers = {p.get("ticker", "") for p in positions}

        for ticker in tickers:
            try:
                is_held = ticker in held_tickers
                current_price = 0.0

                # ── LAYER 1: Classic daily candles (trend baseline) ──────────
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                
                if not is_held:
                    # Check if we already have today's daily data cached
                    if ticker in self._daily_cache and self._daily_cache[ticker].get("date") == today_str:
                        cached = self._daily_cache[ticker]
                        classic = cached["classic"]
                        pro = cached["pro"]
                        trend_score = cached["trend_score"]
                        raw_classic_score = classic.get("score", 50)
                        atr_daily = classic.get("atr", current_price * 0.02)
                    else:
                        # Fetch 60 days of daily candles to build a deep profile for NEW candidates
                        candles = market_service.get_candles(ticker, resolution="D", days_back=60)
                        closes = candles.get("close", [])
                        volumes = candles.get("volume", [])
                        highs = candles.get("high", [])
                        lows = candles.get("low", [])

                        if not closes or len(closes) < 10:
                            logger.warning(f"Insufficient daily data for {ticker}")
                            continue

                        classic = compute_trend_score(closes, volumes, highs, lows)
                        returns = compute_daily_returns(closes)
                        classic["sharpe"] = compute_sharpe_ratio(returns)

                        current_price = closes[-1]
                        atr_daily = classic.get("atr", current_price * 0.02)
                        classic["entry_zone"] = round(current_price - atr_daily * 0.5, 2)
                        classic["exit_zone"]  = round(current_price + atr_daily * 1.5, 2)
                        classic["current_price"] = current_price

                        raw_classic_score = classic.get("score", 50)
                        trend_score = round(raw_classic_score * 0.40, 1)

                        pro = {}
                        pro["trend_bias"] = compute_ema_trend_bias(closes, period=100)
                        pro["vwap"]       = compute_vwap(closes, volumes, highs, lows)
                        pro["rsi_exhaustion"] = detect_rsi_exhaustion(closes)
                        pro["parabolic"]  = detect_parabolic_reversal(closes)
                        if highs and lows:
                            pro["stop_hunt"] = detect_stop_hunt_zones(closes, highs, lows)
                        else:
                            pro["stop_hunt"] = {"stop_hunt_long": False, "stop_hunt_short": False}
                        pro["mm_refill"]   = detect_mm_refill(closes, volumes)
                        pro["first_hour"]  = detect_first_hour_trend(closes)
                        
                        # Save to cache
                        self._daily_cache[ticker] = {
                            "date": today_str,
                            "classic": classic,
                            "pro": pro,
                            "trend_score": trend_score
                        }
                
                else:
                    # Skip 60-day daily fetch for held positions to save API/compute.
                    classic = {"score": 50, "atr": 0.0, "current_price": 0.0, "entry_zone": 0.0, "exit_zone": 0.0, "sharpe": 0.0, "macd": {"trend": "neutral"}}
                    pro = {}
                    raw_classic_score = 50
                    trend_score = 20.0
                    atr_daily = 0.0

                # ── LAYER 2: Intraday quant metrics (execution trigger) ────────
                # Use the exact same precompute pipeline as the backtest:
                #   precompute_quant_metrics_for_ticker → precompute_mtf_tsl_for_ticker → build_quant_metrics
                # This ensures both MTF and VWAP/delta metrics are fully populated.
                try:
                    bars = market_service.get_intraday_bars(ticker, timeframe_minutes=5, days_back=2)
                    if not bars.empty and len(bars) >= 16:
                        df_5m = bars
                        # Step 1: Vectorised ATR / VWAP zscore / delta_ratio / smart_bounce
                        df_precomp = precompute_quant_metrics_for_ticker(df_5m)
                        # Step 2: MTF Trailing SL (4 timeframes: 5m / 120m / 180m / 240m)
                        df_precomp = precompute_mtf_tsl_for_ticker(df_precomp)
                        # Step 3: Build the final quant dict from the precomputed DataFrame
                        quant = build_quant_metrics(df_precomp)
                    else:
                        df_5m = None
                        quant = build_quant_metrics(None)
                except Exception as e:
                    logger.warning(f"Could not compute quant metrics for {ticker}: {e}")
                    df_5m = None
                    quant = build_quant_metrics(None)

                if df_5m is not None and not df_5m.empty:
                    current_price = float(df_5m["close"].iloc[-1])
                    classic["current_price"] = current_price

                # Fallback ATR: use 0.004 * price for 5m (matches backtest + mock_technical_analyst)
                if quant["atr_5m"] == 0.0 and current_price > 0:
                    quant["atr_5m"] = round(atr_daily, 4) if atr_daily > 0 else round(current_price * 0.004, 4)
                    quant["initial_stop"] = round(current_price - 1.5 * quant["atr_5m"], 2)
                    quant["target_2r"] = round(current_price + 3.0 * quant["atr_5m"], 2)

                # ── EXECUTION TRIGGER SCORE (max 60 pts) ─────────────────────
                exec_score = 30  # start neutral at 30/60

                # VWAP Z-score component (±20 pts)
                z = quant.get("vwap_zscore", 0)
                if z < -2.5:
                    exec_score += 20   # statistically oversold → prime buy zone
                elif z < -1.5:
                    exec_score += 12
                elif z < -0.5:
                    exec_score += 6
                elif z > 2.5:
                    exec_score -= 20   # statistically overbought → reject buys
                elif z > 1.5:
                    exec_score -= 10
                elif z > 0.5:
                    exec_score -= 3

                # Volume delta component (±25 pts)
                dr = quant.get("delta_ratio", 0.5)
                if dr > 0.65:
                    exec_score += 25   # strong institutional buying
                elif dr > 0.55:
                    exec_score += 14
                elif dr < 0.35:
                    exec_score -= 25   # strong selling pressure
                elif dr < 0.45:
                    exec_score -= 12

                # Smart bounce bonus (institutional entry signal, +15 pts)
                if quant.get("smart_bounce"):
                    exec_score += 15

                exec_score = max(0, min(60, exec_score))

                # ── COMBINED HYBRID SCORE ─────────────────────────────────────
                hybrid_score = round(trend_score + exec_score)
                hybrid_score = max(0, min(100, hybrid_score))

                # ── SIGNAL LABELS (merged classic + quant) ───────────────────
                signals = []

                # Classic signals from raw score
                if raw_classic_score >= 65:
                    signals.append("STRONG_BUY_TREND")
                elif raw_classic_score >= 50:
                    signals.append("BUY_TREND")
                elif raw_classic_score <= 30:
                    signals.append("STRONG_SELL_TREND")
                elif raw_classic_score <= 40:
                    signals.append("SELL_TREND")
                else:
                    signals.append("HOLD_TREND")

                # RSI signal
                rsi = classic.get("rsi", 50)
                if rsi < 30:
                    signals.append("RSI_OVERSOLD")
                elif rsi > 75:
                    signals.append("RSI_OVERBOUGHT")

                if classic.get("volume_spike"):
                    signals.append("VOLUME_SPIKE")

                macd_trend = classic.get("macd", {}).get("trend", "neutral")
                if macd_trend == "bullish_crossover":
                    signals.append("MACD_BULLISH_CROSS")
                elif macd_trend == "bullish":
                    signals.append("MACD_BULLISH")
                elif macd_trend == "bearish_crossover":
                    signals.append("MACD_BEARISH_CROSS")
                elif macd_trend == "bearish":
                    signals.append("MACD_BEARISH")

                # EMA100 (legacy pro signal kept)
                trend_bias = pro.get("trend_bias", {})
                if trend_bias.get("trend_bias") == "bullish":
                    signals.append("EMA100_BULLISH")
                elif trend_bias.get("trend_bias") == "bearish":
                    signals.append("EMA100_BEARISH")

                # Legacy pro signals
                if pro.get("vwap", {}).get("smart_bounce"):
                    signals.append("VWAP_SMART_BOUNCE")
                rsi_exh = pro.get("rsi_exhaustion", {})
                if rsi_exh.get("setup_active"):
                    if rsi_exh.get("exhaustion") == "overbought_extreme":
                        signals.append("RSI_EXHAUSTION_SHORT")
                    elif rsi_exh.get("exhaustion") == "oversold_extreme":
                        signals.append("RSI_EXHAUSTION_LONG")
                parabolic = pro.get("parabolic", {})
                if parabolic.get("reversal_detected"):
                    signals.append(f"PARABOLIC_{parabolic['direction'].upper()}")
                stop_hunt = pro.get("stop_hunt", {})
                if stop_hunt.get("stop_hunt_long"):
                    signals.append("STOP_HUNT_LONG")
                elif stop_hunt.get("stop_hunt_short"):
                    signals.append("STOP_HUNT_SHORT")
                mm = pro.get("mm_refill", {})
                if mm.get("refill_detected"):
                    signals.append(f"MM_REFILL_{mm['direction_hint'].upper()}")

                # ── Quant signals (new) ───────────────────────────────────────
                signals.extend(quant.get("quant_signals", []))

                # ── QUANT ENTRY SIGNAL GATE (backtest evaluate_bull_entry logic) ──────
                # This matches evaluate_bull_entry() in run_backtest_v2.py exactly:
                #   hybrid_score >= 56  (trend 0-40 + exec 0-60)
                #   delta_ratio > 0.55  (net buying pressure)
                #   vwap_zscore <= 2.5  (not statistically overbought)
                vwap_z_ok = quant.get("vwap_zscore", 0) <= 2.5
                delta_ok = quant.get("delta_ratio", 0.5) > 0.55
                score_ok = hybrid_score >= 56
                quant_entry_signal = score_ok and delta_ok and vwap_z_ok

                # ── Assemble final profile ────────────────────────────────────
                profile = {
                    **classic,
                    "score": hybrid_score,
                    "trend_score": trend_score,
                    "exec_score": exec_score,
                    "quant_entry_signal": quant_entry_signal,   # True = all 3 gates pass
                    "signals": signals,
                    "pro_signals": pro,
                    # Quant fields
                    "atr_5m": quant["atr_5m"],
                    "initial_stop": quant["initial_stop"],
                    "target_2r": quant["target_2r"],
                    "vwap": quant["vwap"],
                    "vwap_zscore": quant["vwap_zscore"],
                    "vwap_signal": quant["vwap_signal"],
                    "price_vs_vwap": quant["price_vs_vwap"],
                    "delta_ratio": quant["delta_ratio"],
                    "buy_pressure": quant["buy_pressure"],
                    "sell_pressure": quant["sell_pressure"],
                    "smart_bounce": quant["smart_bounce"],
                    "last_bar_buy_pct": quant["last_bar_buy_pct"],
                    "mtf_total_pos": quant.get("mtf_total_pos", 0),
                    "tsl_1": quant.get("tsl_1", 0.0),
                }
                profiles[ticker] = profile

                logger.info(
                    f"{ticker}: hybrid={hybrid_score} "
                    f"(trend={trend_score:.0f}/40 + exec={exec_score}/60) | "
                    f"MTF={quant.get('mtf_total_pos', 0)} | δ={quant['delta_ratio']:.2f} | "
                    f"ATR=${quant['atr_5m']:.3f}"
                )

            except Exception as e:
                logger.error(f"Technical analysis failed for {ticker}: {e}")

        logger.info(f"TechnicalAnalyst: analyzed {len(profiles)} tickers")
        return profiles
