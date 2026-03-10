import os
import sys
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_market_service import MockMarketService
from backtest.mock_technical_analyst import MockTechnicalAnalyst
from backtest.data_loader import TICKERS

LOG_FILE = os.path.join(os.path.dirname(__file__), "results.log")

def run_backtest(params=None, quiet=False, data_file=None, daily_file=None, market_service=None):
    if params is None:
        params = {
            "long_hybrid": 70, "long_exec": 40, "long_vwap": -1.5,
            "short_hybrid": 30, "short_exec": 20, "short_vwap": 1.5,
            "stop_r": 1.5, "target_r": 3.0,
            "trail_1r": True, "trail_2r": True, 
            "trail_activation_r": 1.0, "trailing_distance": 1.5,
            "use_mtf": False
        }
    
    if market_service is not None:
        # Fast path: reuse pre-loaded data (no disk I/O)
        all_times = set()
        for ticker, group in market_service._intraday_by_ticker.items():
            all_times.update(group.index.tolist())
        unique_times = sorted(all_times)
        start_time = unique_times[len(unique_times) // 4]
        times_to_simulate = [t for t in unique_times if t >= start_time]
        
        if not quiet:
            print(f"Starting Backtest on {len(TICKERS)} tickers across {len(times_to_simulate)} 5-min intervals")
        
        # Reset market service state for this backtest run
        market_service.set_time(times_to_simulate[0])
    else:
        # Original path: load from disk
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        hist_path = data_file if data_file else os.path.join(data_dir, "historical_bars.csv")
        day_path = daily_file if daily_file else os.path.join(data_dir, "daily_bars.csv")
        
        if not os.path.exists(hist_path) or not os.path.exists(day_path):
            print(f"Data files not found. Please run data_loader.py first.")
            return
    
        if not quiet:
            print("Loading data into memory...")
        df_intraday = pd.read_csv(hist_path)
        df_intraday["timestamp"] = pd.to_datetime(df_intraday["timestamp"], utc=True)
        
        unique_times = df_intraday["timestamp"].drop_duplicates().sort_values().tolist()
        start_time = unique_times[len(unique_times) // 4]
        times_to_simulate = [t for t in unique_times if t >= start_time]
    
        if not quiet:
            print(f"Starting Backtest on {len(TICKERS)} tickers across {len(times_to_simulate)} 5-min intervals")
        
        market_service = MockMarketService(current_time=times_to_simulate[0], data_file=data_file, daily_file=daily_file)
    ta = MockTechnicalAnalyst(mock_market_service=market_service)

    positions = []
    trade_history = []
    
    if not quiet:
        with open(LOG_FILE, "w") as f:
            f.write(f"--- 5-Minute Technical Analysis Backtest Log ---\n")
            f.write(f"Universe: {len(TICKERS)} tickers\n")
            f.write(f"Intervals: {len(times_to_simulate)}\n\n")
        
    def log_trade(msg):
        if not quiet:
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")

    step = 0
    for current_timestamp in times_to_simulate:
        step += 1
        market_service.set_time(current_timestamp)
        
        # 1. Manage open positions (Check stop/target on current bar)
        for pos in list(positions):
            quote = market_service.get_quote(pos["ticker"])
            if not quote: continue
            
            curr_price = float(quote.get("current", 0.0))
            if curr_price == 0: continue
            
            is_long = pos.get("side", "long") == "long"
            
            if is_long:
                # 1. Trailing Runner (+2R threshold activates pure trailing)
                if params.get("trail_2r", True) and curr_price >= pos["entry_price"] + 2.0 * pos["atr_5m"]:
                    new_stop = curr_price - params.get("trailing_distance", 1.5) * pos["atr_5m"]
                    if new_stop > pos["stop"]:
                        pos["stop"] = new_stop
                        if pos["target"] != float('inf'):
                            pos["target"] = float('inf') # Remove hard target
                            log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (LONG) Runner Activated! Target removed.")
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (LONG) Trailing Stop shifted to @ {pos['stop']:.2f}")

                # 2. Trailing Stop to Breakeven (Activation threshold)
                elif params.get("trail_1r", True) and curr_price >= pos["entry_price"] + params.get("trail_activation_r", 1.0) * pos["atr_5m"]:
                    if pos["stop"] < pos["entry_price"]:
                        pos["stop"] = pos["entry_price"]
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (LONG) Trailing Stop moved to Breakeven @ {pos['stop']:.2f}")
            else:
                # 1. Trailing Runner (+2R threshold activates pure trailing)
                if params.get("trail_2r", True) and curr_price <= pos["entry_price"] - 2.0 * pos["atr_5m"]:
                    new_stop = curr_price + params.get("trailing_distance", 1.5) * pos["atr_5m"]
                    if new_stop < pos["stop"]:
                        pos["stop"] = new_stop
                        if pos["target"] != -float('inf'):
                            pos["target"] = -float('inf') # Remove hard target
                            log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (SHORT) Runner Activated! Target removed.")
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (SHORT) Trailing Stop shifted to @ {pos['stop']:.2f}")

                # 2. Trailing Stop to Breakeven (Activation threshold)
                elif params.get("trail_1r", True) and curr_price <= pos["entry_price"] - params.get("trail_activation_r", 1.0) * pos["atr_5m"]:
                    if pos["stop"] > pos["entry_price"]:
                        pos["stop"] = pos["entry_price"]
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {pos['ticker']} (SHORT) Trailing Stop moved to Breakeven @ {pos['stop']:.2f}")

            # Check Stop Loss / Target / MTF
            pnl_pct = 0
            closed = False
            close_type = ""
            
            # 1. Raw asset price movement
            if is_long:
                asset_pnl_pct = (curr_price - pos["entry_price"]) / pos["entry_price"]
            else:
                asset_pnl_pct = (pos["entry_price"] - curr_price) / pos["entry_price"]
                
            # 2. Capital Compounding - apply leverage calculated at entry
            leverage = pos.get("leverage", 1.0)
            pnl_pct = (asset_pnl_pct * leverage) * 100
            
            # 3. Check exit triggers
            if params.get("use_mtf", False) and pos.get("force_close"):
                closed = True
                close_type = "mtf_tsl"
            elif not params.get("use_mtf", False):
                if is_long:
                    if curr_price <= pos["stop"]:
                        closed = True
                        close_type = "stop"
                    elif curr_price >= pos["target"]:
                        closed = True
                        close_type = "target"
                else:
                    if curr_price >= pos["stop"]:
                        closed = True
                        close_type = "stop"
                    elif curr_price <= pos["target"]:
                        closed = True
                        close_type = "target"
                    
            if closed:
                action = "STOP OUT" if close_type == "stop" else "TARGET REACHED" if close_type == "target" else "MTF EXIT"
                side_str = "LONG" if is_long else "SHORT"
                log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {action} {pos['ticker']} ({side_str}) @ {curr_price:.2f} | PnL: {pnl_pct:.2f}%")
                trade_history.append({"ticker": pos["ticker"], "pnl": pnl_pct, "type": close_type})
                positions.remove(pos)

        # 2. Scan for new entries & update MTF holding states
        if True: # Evaluate every 5 mins for precision MTF exits
            profiles = ta.analyze(TICKERS, positions=[{"ticker": p["ticker"]} for p in positions])
            
            for ticker, profile in profiles.items():
                mtf_total_pos = profile.get("mtf_total_pos", 0)
                
                # MTF Trailing Exit evaluation for held positions
                held_pos = next((p for p in positions if p["ticker"] == ticker), None)
                if held_pos and params.get("use_mtf", False):
                    is_long = held_pos["side"] == "long"
                    if (is_long and mtf_total_pos <= 0) or (not is_long and mtf_total_pos >= 0):
                        held_pos["force_close"] = True
                        continue

                # Skip if already held (for entries)
                if held_pos:
                    continue
                
                hybrid_score = profile.get("score", 50)
                vwap_zscore = profile.get("vwap_zscore", 0)
                exec_score = profile.get("exec_score", 30)
                delta_ratio = profile.get("delta_ratio", 0.5)
                smart_bounce = profile.get("smart_bounce", False)
                last_bar_buy_pct = profile.get("last_bar_buy_pct", 0.5)
                signals = profile.get("signals", [])
                
                entry_price = profile.get("current_price", 0)
                if entry_price == 0: continue
                
                # Evaluate MTF entry conditions if enabled
                mtf_ok_long = mtf_total_pos == 4 if params.get("use_mtf", False) else True
                mtf_ok_short = mtf_total_pos == -4 if params.get("use_mtf", False) else True

                # Determine strategy type for differentiated entry logic
                strategy_type = params.get("strategy_type", "trend")
                
                long_entry = False
                short_entry = False
                
                if strategy_type == "trend":
                    # === TREND FOLLOWER (Loose Entry) ===
                    # Original logic: strong trend + good execution + favorable VWAP
                    long_entry = (mtf_ok_long 
                                  and hybrid_score >= params["long_hybrid"] 
                                  and exec_score >= params["long_exec"] 
                                  and vwap_zscore < params["long_vwap"])
                    short_entry = (mtf_ok_short 
                                   and hybrid_score <= params["short_hybrid"] 
                                   and exec_score <= params["short_exec"] 
                                   and vwap_zscore > params["short_vwap"])
                    
                elif strategy_type == "strict":
                    # === STRICT EXECUTION (Momentum Breakout) ===
                    # Requires confirmed institutional buying pressure (delta_ratio)
                    # + strong hybrid score + solid exec score
                    # VWAP is NOT used as filter — this strategy enters on strength, not dips
                    long_entry = (mtf_ok_long 
                                  and hybrid_score >= params["long_hybrid"] 
                                  and exec_score >= params["long_exec"] 
                                  and delta_ratio >= params.get("long_delta_min", 0.58))
                    short_entry = (mtf_ok_short 
                                   and hybrid_score <= params["short_hybrid"] 
                                   and exec_score <= params["short_exec"] 
                                   and delta_ratio <= params.get("short_delta_max", 0.42))
                    
                elif strategy_type == "pullback":
                    # === PULLBACK BUYER (Confirmed Mean-Reversion) ===
                    # Requires VWAP oversold condition + bounce confirmation
                    # (smart_bounce OR high last_bar_buy_pct = evidence of reversal)
                    # Hybrid score floor is low — trend doesn't need to be strong for dip-buying
                    bounce_confirmed_long = (smart_bounce or last_bar_buy_pct >= params.get("long_bounce_pct", 0.62))
                    bounce_confirmed_short = (last_bar_buy_pct <= params.get("short_bounce_pct", 0.38))
                    
                    long_entry = (mtf_ok_long 
                                  and vwap_zscore < params["long_vwap"]
                                  and bounce_confirmed_long
                                  and exec_score >= params.get("long_exec", 20))
                    short_entry = (mtf_ok_short 
                                   and vwap_zscore > params["short_vwap"]
                                   and bounce_confirmed_short
                                   and exec_score <= params.get("short_exec", 40))

                # Execute entry
                if long_entry:
                    atr_5m = profile.get("atr_5m", 0)
                    stop = entry_price - params.get("stop_r", 1.5) * atr_5m
                    target = entry_price + params.get("target_r", 3.0) * atr_5m
                    
                    if stop > 0 and target > 0 and stop < entry_price:
                        # Capital Compounding: Risk 5% of equity per trade
                        risk_pct = params.get("risk_per_trade_pct", 0.05)
                        stop_dist_pct = abs(entry_price - stop) / entry_price
                        leverage = min(risk_pct / stop_dist_pct, 10.0) if stop_dist_pct > 0 else 1.0
                        
                        pos = {
                            "ticker": ticker, "side": "long", "entry_time": current_timestamp,
                            "entry_price": entry_price, "stop": stop, "target": target,
                            "atr_5m": atr_5m, "force_close": False, "leverage": leverage
                        }
                        positions.append(pos)
                        tag = "[MTF] " if params.get("use_mtf", False) else ""
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {tag}BUY {ticker} @ {entry_price:.2f} | Score: {hybrid_score} | Lev: {leverage:.1f}x | Stop: {stop:.2f} | Tgt: {target:.2f}")
                
                elif short_entry:
                    atr_5m = profile.get("atr_5m", 0)
                    stop = entry_price + params.get("stop_r", 1.5) * atr_5m
                    target = entry_price - params.get("target_r", 3.0) * atr_5m
                    
                    if stop > 0 and target > 0 and stop > entry_price:
                        # Capital Compounding: Risk 5% of equity per trade
                        risk_pct = params.get("risk_per_trade_pct", 0.05)
                        stop_dist_pct = abs(entry_price - stop) / entry_price
                        leverage = min(risk_pct / stop_dist_pct, 10.0) if stop_dist_pct > 0 else 1.0
                        
                        pos = {
                            "ticker": ticker, "side": "short", "entry_time": current_timestamp,
                            "entry_price": entry_price, "stop": stop, "target": target,
                            "atr_5m": atr_5m, "force_close": False, "leverage": leverage
                        }
                        positions.append(pos)
                        tag = "[MTF] " if params.get("use_mtf", False) else ""
                        log_trade(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] {tag}SHORT {ticker} @ {entry_price:.2f} | Score: {hybrid_score} | Lev: {leverage:.1f}x | Stop: {stop:.2f} | Tgt: {target:.2f}")

    # End of backtest
    wins = len([t for t in trade_history if t["pnl"] > 0])
    losses = len([t for t in trade_history if t["pnl"] <= 0])
    gross_profit = sum(t["pnl"] for t in trade_history if t["pnl"] > 0)
    gross_loss = sum(abs(t["pnl"]) for t in trade_history if t["pnl"] <= 0)
    total_trades = len(trade_history)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = sum(t["pnl"] for t in trade_history) / total_trades if total_trades > 0 else 0

    summary = f"\n=== BACKTEST COMPLETE ==="
    summary += f"\nTotal Trades: {total_trades}"
    summary += f"\nWins: {wins} | Losses: {losses} | Win Rate: {win_rate:.2f}%"
    summary += f"\nAverage Return Per Trade: {avg_pnl:.2f}%"
    
    
    log_trade(summary)
    
    return {
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss
    }

if __name__ == "__main__":
    run_backtest()
