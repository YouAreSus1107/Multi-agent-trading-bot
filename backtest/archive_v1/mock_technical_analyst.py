import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STRATEGY_CONFIG
from utils.indicators import (
    compute_trend_score, compute_daily_returns, compute_sharpe_ratio,
    compute_ema_trend_bias, compute_vwap, detect_rsi_exhaustion,
    detect_parabolic_reversal, detect_stop_hunt_zones,
    detect_mm_refill, detect_first_hour_trend,
)
from utils.quant_engine import build_quant_metrics
from utils.logger import get_logger

logger = get_logger("mock_technical_analyst")

class MockTechnicalAnalyst:
    def __init__(self, mock_market_service):
        self._daily_cache = {}
        self.market_service = mock_market_service

    def analyze(self, tickers: list[str], positions: list[dict] = None) -> dict:
        profiles = {}
        if positions is None:
            positions = []
        held_tickers = {p.get("ticker", "") for p in positions}

        for ticker in tickers:
            try:
                is_held = ticker in held_tickers
                current_price = 0.0

                today_str = self.market_service.current_time.strftime("%Y-%m-%d")
                
                if not is_held:
                    if ticker in self._daily_cache and self._daily_cache[ticker].get("date") == today_str:
                        cached = self._daily_cache[ticker]
                        classic = cached["classic"]
                        pro = cached["pro"]
                        trend_score = cached["trend_score"]
                        raw_classic_score = classic.get("score", 50)
                        atr_daily = classic.get("atr", current_price * 0.02)
                    else:
                        candles = self.market_service.get_candles(ticker, resolution="D", days_back=60)
                        closes = candles.get("close", [])
                        volumes = candles.get("volume", [])
                        highs = candles.get("high", [])
                        lows = candles.get("low", [])

                        if not closes or len(closes) < 10:
                            logger.debug(f"Insufficient daily data for {ticker}")
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
                        
                        self._daily_cache[ticker] = {
                            "date": today_str,
                            "classic": classic,
                            "pro": pro,
                            "trend_score": trend_score
                        }
                else:
                    classic = {"score": 50, "atr": 0.0, "current_price": 0.0, "entry_zone": 0.0, "exit_zone": 0.0, "sharpe": 0.0, "macd": {"trend": "neutral"}}
                    pro = {}
                    raw_classic_score = 50
                    trend_score = 20.0
                    atr_daily = 0.0

                # The Universe Filter (40-60 score early return) was removed here because execution metrics
                # like VWAP, ATR, and Delta are now precomputed in O(1) so we must evaluate everything
                # to catch mean-reversion anomalies missed by the classic trend algorithm.

                # Fast path: get just the single latest bar via O(log n) binary search
                # instead of slicing a 20-day DataFrame (eliminates 58,500 DataFrame copies per dataset)
                latest_bar = self.market_service.get_latest_bar(ticker)
                
                if latest_bar is None:
                    continue
                
                # Extract precomputed MTF values directly from the Series (O(1))
                mtf_total_pos = int(latest_bar.get("mtf_total_pos", 0)) if "mtf_total_pos" in latest_bar.index else 0
                tsl_1 = float(latest_bar.get("tsl_1", 0.0)) if "tsl_1" in latest_bar.index else 0.0
                
                # Extract precomputed quant metrics directly (no build_quant_metrics call needed)
                current_price = float(latest_bar["close"])
                classic["current_price"] = current_price
                
                atr_val = float(latest_bar.get("atr_5m", 0.0)) if "atr_5m" in latest_bar.index else 0.0
                vwap_val = float(latest_bar.get("vwap", 0.0)) if "vwap" in latest_bar.index else 0.0
                vwap_z = float(latest_bar.get("vwap_zscore", 0.0)) if "vwap_zscore" in latest_bar.index else 0.0
                sigma_val = float(latest_bar.get("sigma", 0.0)) if "sigma" in latest_bar.index else 0.0
                dr = float(latest_bar.get("delta_ratio", 0.5)) if "delta_ratio" in latest_bar.index else 0.5
                bp = float(latest_bar.get("buy_pressure", 0)) if "buy_pressure" in latest_bar.index else 0
                sp = float(latest_bar.get("sell_pressure", 0)) if "sell_pressure" in latest_bar.index else 0
                sb = bool(latest_bar.get("smart_bounce", False)) if "smart_bounce" in latest_bar.index else False
                lbbp = float(latest_bar.get("last_bar_buy_pct", 0.5)) if "last_bar_buy_pct" in latest_bar.index else 0.5
                
                if vwap_z > 2.5:
                    vwap_signal = "overbought"
                elif vwap_z < -2.5:
                    vwap_signal = "oversold"
                else:
                    vwap_signal = "neutral"
                
                quant_signals = []
                if vwap_z > 2.5: quant_signals.append("VWAP_OVERBOUGHT")
                elif vwap_z < -2.5: quant_signals.append("VWAP_OVERSOLD")
                elif abs(vwap_z) < 1.0: quant_signals.append("VWAP_NEUTRAL")
                if sb: quant_signals.append("SMART_BOUNCE")
                if dr > 0.55: quant_signals.append("VOLUME_BULLISH")
                elif dr < 0.45: quant_signals.append("VOLUME_BEARISH")
                
                # Build the quant dict directly (replaces build_quant_metrics call)
                quant = {
                    "atr_5m": atr_val,
                    "vwap": vwap_val,
                    "vwap_zscore": vwap_z,
                    "vwap_signal": vwap_signal,
                    "price_vs_vwap": "above" if current_price > vwap_val else "below",
                    "delta_ratio": dr,
                    "buy_pressure": bp,
                    "sell_pressure": sp,
                    "smart_bounce": sb,
                    "last_bar_buy_pct": lbbp,
                    "quant_signals": quant_signals,
                }
                
                # ATR fallback for edge cases
                if quant["atr_5m"] == 0.0 and current_price > 0:
                    quant["atr_5m"] = round(atr_daily, 4) if atr_daily > 0 else round(current_price * 0.015, 4)

                quant["long_stop"] = round(current_price - 1.5 * quant["atr_5m"], 2)
                quant["long_target"] = round(current_price + 3.0 * quant["atr_5m"], 2)
                quant["short_stop"] = round(current_price + 1.5 * quant["atr_5m"], 2)
                quant["short_target"] = round(current_price - 3.0 * quant["atr_5m"], 2)

                exec_score = 30
                z = quant.get("vwap_zscore", 0)
                if z < -2.5: exec_score += 20
                elif z < -1.5: exec_score += 12
                elif z < -0.5: exec_score += 6
                elif z > 2.5: exec_score -= 20
                elif z > 1.5: exec_score -= 10
                elif z > 0.5: exec_score -= 3

                dr = quant.get("delta_ratio", 0.5)
                if dr > 0.65: exec_score += 25
                elif dr > 0.55: exec_score += 14
                elif dr < 0.35: exec_score -= 25
                elif dr < 0.45: exec_score -= 12

                if quant.get("smart_bounce"):
                    exec_score += 15

                exec_score = max(0, min(60, exec_score))
                hybrid_score = round(trend_score + exec_score)
                hybrid_score = max(0, min(100, hybrid_score))

                signals = []
                if raw_classic_score >= 65: signals.append("STRONG_BUY_TREND")
                elif raw_classic_score >= 50: signals.append("BUY_TREND")
                elif raw_classic_score <= 30: signals.append("STRONG_SELL_TREND")
                elif raw_classic_score <= 40: signals.append("SELL_TREND")
                else: signals.append("HOLD_TREND")

                rsi = classic.get("rsi", 50)
                if rsi < 30: signals.append("RSI_OVERSOLD")
                elif rsi > 75: signals.append("RSI_OVERBOUGHT")

                if classic.get("volume_spike"): signals.append("VOLUME_SPIKE")

                macd_trend = classic.get("macd", {}).get("trend", "neutral")
                if macd_trend == "bullish_crossover": signals.append("MACD_BULLISH_CROSS")
                elif macd_trend == "bullish": signals.append("MACD_BULLISH")
                elif macd_trend == "bearish_crossover": signals.append("MACD_BEARISH_CROSS")
                elif macd_trend == "bearish": signals.append("MACD_BEARISH")

                trend_bias = pro.get("trend_bias", {})
                if trend_bias.get("trend_bias") == "bullish": signals.append("EMA100_BULLISH")
                elif trend_bias.get("trend_bias") == "bearish": signals.append("EMA100_BEARISH")

                if pro.get("vwap", {}).get("smart_bounce"): signals.append("VWAP_SMART_BOUNCE")
                rsi_exh = pro.get("rsi_exhaustion", {})
                if rsi_exh.get("setup_active"):
                    if rsi_exh.get("exhaustion") == "overbought_extreme": signals.append("RSI_EXHAUSTION_SHORT")
                    elif rsi_exh.get("exhaustion") == "oversold_extreme": signals.append("RSI_EXHAUSTION_LONG")
                parabolic = pro.get("parabolic", {})
                if parabolic.get("reversal_detected"): signals.append(f"PARABOLIC_{parabolic['direction'].upper()}")
                stop_hunt = pro.get("stop_hunt", {})
                if stop_hunt.get("stop_hunt_long"): signals.append("STOP_HUNT_LONG")
                elif stop_hunt.get("stop_hunt_short"): signals.append("STOP_HUNT_SHORT")
                mm = pro.get("mm_refill", {})
                if mm.get("refill_detected"): signals.append(f"MM_REFILL_{mm['direction_hint'].upper()}")

                signals.extend(quant.get("quant_signals", []))

                profile = {
                    **classic,
                    "score": hybrid_score,
                    "trend_score": trend_score,
                    "exec_score": exec_score,
                    "signals": signals,
                    "pro_signals": pro,
                    "atr_5m": quant.get("atr_5m", 0.0),
                    "long_stop": quant["long_stop"],
                    "long_target": quant["long_target"],
                    "short_stop": quant["short_stop"],
                    "short_target": quant["short_target"],
                    "vwap": quant["vwap"],
                    "vwap_zscore": quant["vwap_zscore"],
                    "vwap_signal": quant["vwap_signal"],
                    "price_vs_vwap": quant["price_vs_vwap"],
                    "delta_ratio": quant["delta_ratio"],
                    "buy_pressure": quant["buy_pressure"],
                    "sell_pressure": quant["sell_pressure"],
                    "smart_bounce": quant["smart_bounce"],
                    "last_bar_buy_pct": quant["last_bar_buy_pct"],
                    "mtf_total_pos": mtf_total_pos,
                    "tsl_1": tsl_1
                }
                profiles[ticker] = profile

            except Exception as e:
                logger.error(f"Mock Technical analysis failed for {ticker}: {e}")

        return profiles
