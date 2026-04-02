"""
optimize_v2.py — Smart Overnight Parameter Optimizer for run_backtest_v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm: Bayesian Optimization via Optuna TPE (Tree-structured Parzen Estimator)
Unlike random search, TPE builds a probabilistic model of which parameter
regions produce high scores, then ACTIVELY samples from promising regions.

Usage:
  python optimize_v2.py                  # Start fresh or resume from SQLite
  python optimize_v2.py --resume         # Explicitly resume from previous session
  python optimize_v2.py --n-jobs 2       # Run 2 parallel workers on the same study
  python optimize_v2.py --show-best 20   # Show the top-20 results and exit
  python optimize_v2.py --export-best    # Export best params to test_params_optimized.txt

Press Ctrl+C to stop cleanly at any time. Progress is saved after every trial.
"""

import os
import sys
import csv
import time
import json
import sqlite3
import argparse
import logging
import threading
from datetime import datetime
from copy import deepcopy

import optuna
import pandas as pd

# ── Path Setup ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

# Silence verbose backtest logs inside optimizer
os.environ["DISABLE_FILE_LOGGING"] = "1"

from backtest.run_backtest_v2 import run_backtest_v2
from services.discord_notifier import send_optimizer_best

# Also silence Optuna's own verbose logging (we print our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
STUDY_DB   = os.path.join(THIS_DIR, "optimize_v2_study.db")
RESULTS_CSV = os.path.join(THIS_DIR, "v2_optimization_results.csv")
BEST_PARAMS_OUT = os.path.join(THIS_DIR, "test_params.txt")

# Lock serialising CSV writes — required when n_jobs > 1 spawns multiple threads
_csv_lock = threading.Lock()

# ── Multi-period evaluation windows ──────────────────────────────────────────
# Strategies must perform well across ALL regimes to score high.
EVAL_PERIODS = [
    {"name": "Jan-Mar 2021  (Post-Vaccine Bull Melt-Up)",  "start": "2021-01-01", "end": "2021-03-31"},
    {"name": "Aug-Oct 2022  (Hard Bear Market)",           "start": "2022-08-01", "end": "2022-10-31"},
    {"name": "May-Jul 2023  (AI Melt-Up Bull)",            "start": "2023-05-01", "end": "2023-07-31"},
    {"name": "Oct-Dec 2021  (Late-Cycle Choppy Bull)",     "start": "2021-10-01", "end": "2021-12-31"},
]

# ── Warm-start seeds (enqueue these as the first Optuna trials) ──────────────
WARM_START_SEEDS = [
    # Baseline — moderate thresholds across both strategies
    {
        "mom_vwap_z_min": 0.3, "mom_vol_ratio_min": 1.3, "mom_delta_min": 0.58,
        "rev_vwap_z_max": -2.5, "rev_vol_spike_min": 2.0,
        "stop_r": 1.5, "target_r": 2.0, "risk_per_trade": 0.05,
    },
    # Selective — tighter entries, wider stops
    {
        "mom_vwap_z_min": 0.5, "mom_vol_ratio_min": 1.5, "mom_delta_min": 0.62,
        "rev_vwap_z_max": -3.0, "rev_vol_spike_min": 2.5,
        "stop_r": 2.0, "target_r": 2.5, "risk_per_trade": 0.04,
    },
    # Permissive — looser entries, tighter stops
    {
        "mom_vwap_z_min": 0.2, "mom_vol_ratio_min": 1.2, "mom_delta_min": 0.56,
        "rev_vwap_z_max": -2.0, "rev_vol_spike_min": 1.5,
        "stop_r": 1.25, "target_r": 2.0, "risk_per_trade": 0.06,
    },
]

# ── CSV Header ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "trial_number", "composite_score",
    "avg_profit_factor", "avg_win_rate", "avg_drawdown", "avg_pnl", "avg_total_return", "total_trades",
    "mom_vwap_z_min", "mom_vol_ratio_min", "mom_delta_min",
    "rev_vwap_z_max", "rev_vol_spike_min",
    "stop_r", "target_r", "risk_per_trade",
]


def _cache_has_data_for_period(start: str, end: str, min_files: int = 50) -> bool:
    """Return True if enough 5m cache parquet files exist for this date range."""
    from backtest.data_loader_v2 import CACHE_DIR
    if not os.path.isdir(CACHE_DIR):
        return False
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    count = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".parquet"):
            continue
        # filename: TICKER_YYYY-MM-DD.parquet  (split on last underscore)
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            file_date = pd.Timestamp(parts[1].replace(".parquet", ""))
            if start_ts <= file_date <= end_ts:
                count += 1
                if count >= min_files:
                    return True
        except Exception:
            continue
    return False


def _prefetch_eval_data() -> None:
    """
    Automatically fetch and cache 5m bar data for all eval periods before the
    first trial. Only downloads periods that are missing or sparse on disk.
    Runs once; all subsequent trials use the disk cache with fetch_5m=False.
    """
    missing = [p for p in EVAL_PERIODS if not _cache_has_data_for_period(p["start"], p["end"])]
    if not missing:
        return

    print(f"\n  [Prefetch] {len(missing)} eval period(s) missing 5m data — fetching now.")
    print(f"  [Prefetch] This runs once and caches everything to disk.\n")

    # Minimal params — we only care about populating the cache, not the trade results.
    _fetch_params = {
        'stop_r': 3.0, 'target_r': 3.5, 'risk_per_trade': 0.05,
        'target_trade_vol': 2.0, 'max_total_leverage': 1.5,
        'max_positions': 5, 'max_hold_trading_days': 10,
        'ticker_cooldown_days': 3, 'loss_cooldown_threshold': -1.5,
        'ticker_cooldown_extended_days': 10, 'slippage_pct': 0.0003,
        'commission_pct': 0.0,
    }
    for period in missing:
        print(f"  [Prefetch] Fetching: {period['name']}  ({period['start']} → {period['end']}) ...")
        try:
            run_backtest_v2(
                start_date=period["start"],
                end_date=period["end"],
                fetch_5m=True,
                verbose=False,
                params=_fetch_params,
            )
            print(f"  [Prefetch] Done: {period['name']}")
        except Exception as e:
            print(f"  [Prefetch] WARNING: prefetch failed for {period['name']}: {e}")

    print(f"\n  [Prefetch] All eval periods cached. Starting optimization...\n")


def compute_composite_score(results: list[dict]) -> float:
    """
    Compute the composite score across all evaluation periods.

    Components (each period) — rebalanced to reward cumulative edge, not just PF:
      - Profit Factor   × 0.15   (↓ from 0.40 — noisy at low N, redundant w/ avg_pnl)
      - Win Rate (0-1)  × 0.15   (↓ from 0.30)
      - avg_pnl (×2)    × 0.25   (↑ from 0.15 — direct per-trade edge measure)
      - total_return     × 0.30   (NEW — cumulative edge = avg_pnl × opportunity count)
      - 1/max_drawdown   × 0.15   (unchanged)

    The avg_pnl and total_return components are in natural tension:
      avg_pnl rewards SELECTIVITY (few high-quality trades).
      total_return rewards THROUGHPUT (enough trades to compound gains).
      A strategy needs BOTH — the optimizer can't game one without the other.

    Trade count handling (raised floor + soft bonus):
      - 0 trades          → score = 0.0  (hard zero)
      - 1–39 trades       → score × (trades / 40)  smooth penalty ramp
      - 40–99 trades      → soft bonus, linearly scaling to +20%
      - 100+ trades       → capped at +20% bonus

    Hard penalties:
      - any period drawdown > 40%   → ×0.2   (blowup risk)
      - profit_factor < 0.8         → ×0.3   (must not hemorrhage in any regime)

    Final: 50% avg + 50% worst - 15% std (cross-year consistency).
    """
    import math
    if not results:
        return 0.0

    total_trades_all = sum(r.get("total_trades", 0) for r in results)

    # Hard early exit: no trades at all → no signal whatsoever
    if total_trades_all == 0:
        return 0.0

    period_scores = []

    for r in results:
        pf     = r.get("profit_factor", 0.0)
        wr     = r.get("win_rate", 0.0) / 100.0
        dd     = abs(r.get("max_drawdown", 50.0))
        pnl    = r.get("avg_pnl", 0.0)
        ret    = r.get("total_return", 0.0)
        trades = r.get("total_trades", 0)

        # Cap PF to prevent inf from blowing up the score (0 losses = inf PF)
        pf_capped = 5.0 if pf == float('inf') else min(pf, 5.0)

        # Component scaling:
        # - avg_pnl (typically 0.1-0.8%) → ×2 to make it 0.2-1.6 range
        # - total_return (typically 5-30% per 3mo period) → /20 to make it 0.25-1.5 range
        # These now have comparable magnitude to PF (1.5-5.0 range).
        dd_component  = 1.0 / max(dd, 1.0)
        pnl_scaled    = pnl * 2.0
        ret_scaled    = ret / 20.0

        score = (pf_capped   * 0.15) \
              + (wr          * 0.15) \
              + (pnl_scaled  * 0.25) \
              + (ret_scaled  * 0.30) \
              + (dd_component * 0.15)

        # Trade count: floor raised to 40 (3-month periods need ~13 trades/month minimum).
        # Below 40: smooth penalty ramp. Above 40: soft bonus for finding more opportunities.
        # The bonus creates a gradient so the optimizer prefers strategies that can SCALE —
        # not just cherry-pick 32 perfect trades per period.
        if trades == 0:
            score = 0.0
        elif trades < 40:
            score *= max(0.1, trades / 40.0)
        else:
            # Soft bonus: 40 trades = 1.0x, 70 trades = 1.10x, 100+ = 1.20x (capped)
            trade_bonus = 1.0 + 0.20 * min((trades - 40) / 60.0, 1.0)
            score *= trade_bonus

        if dd > 40.0:   score *= 0.2
        if pf < 0.8:    score *= 0.3

        period_scores.append(score)

    # Scoring: 50% avg + 50% worst - 15% std
    # The heavy worst-period weight and variance penalty mean a trial scoring
    # [2.0, 2.0, 0.1, 0.1] (0.82) loses decisively to [1.0, 1.0, 1.0, 1.0] (1.0).
    # "All periods good" beats "a few periods extraordinary".
    avg   = sum(period_scores) / len(period_scores)
    worst = min(period_scores)
    n     = len(period_scores)
    std   = math.sqrt(sum((s - avg) ** 2 for s in period_scores) / n)
    return 0.50 * avg + 0.50 * worst - 0.15 * std


def run_trial_across_periods(sparm: dict, verbose: bool = False) -> tuple[float, list[dict]]:
    """Run the backtest across all evaluation periods and return (composite_score, per_period_results)."""
    all_results = []
    for period in EVAL_PERIODS:
        params = {
            "stop_r":                        sparm["stop_r"],
            "target_r":                      sparm["target_r"],
            "risk_per_trade":                sparm.get("risk_per_trade", 0.05),
            "max_positions":                 5,
            "max_hold_trading_days":         10,       # fixed constant
            "ticker_cooldown_days":          3,        # fixed constant
            "loss_cooldown_threshold":       -1.5,     # fixed constant
            "ticker_cooldown_extended_days": 10,       # fixed constant
            "slippage_pct":                  0.0003,   # 3 bps per leg
            "commission_pct":                0.0,      # Alpaca: $0 commissions
        }

        try:
            result = run_backtest_v2(
                start_date=period["start"],
                end_date=period["end"],
                top_n=10,
                fetch_5m=False,          # OPTIMIZER: use disk cache only, never hit Alpaca API live
                verbose=False,
                always_long=False,
                params=params,
                sparm=sparm,
                regime_dwell_days=3,     # Match the dwell setting used in full backtests
            )
        except Exception as exc:
            if verbose:
                print(f"    [!] Period {period['name']} CRASHED: {exc}")
            result = {}

        all_results.append({
            "period": period["name"],
            "total_trades":   result.get("total_trades", 0),
            "win_rate":       result.get("win_rate", 0.0),
            "avg_pnl":        result.get("avg_pnl", 0.0),
            "profit_factor":  result.get("profit_factor", 0.0),
            "max_drawdown":   abs(result.get("max_drawdown", 100.0)),
            "final_equity":   result.get("final_equity", 10000.0),
            "total_return":   result.get("total_return", 0.0),
        })

    composite = compute_composite_score(all_results)
    return composite, all_results


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest parameters, run backtests, return composite score."""

    # ── Parameter Search Space (8 params, down from 17) ────────────────────
    # Momentum: VWAP breakout with volume expansion
    mom_vwap_z_min    = trial.suggest_float("mom_vwap_z_min",     0.1, 1.5, step=0.1)
    mom_vol_ratio_min = trial.suggest_float("mom_vol_ratio_min",  1.1, 2.0, step=0.1)
    mom_delta_min     = trial.suggest_float("mom_delta_min",      0.55, 0.70, step=0.01)

    # Mean reversion: confirmed capitulation + reversal bar
    rev_vwap_z_max    = trial.suggest_float("rev_vwap_z_max",    -3.5, -2.0, step=0.1)
    rev_vol_spike_min = trial.suggest_float("rev_vol_spike_min",  1.5, 3.0, step=0.1)

    # Risk / exit
    stop_r            = trial.suggest_float("stop_r",             1.0, 2.5, step=0.25)
    target_r          = trial.suggest_float("target_r",           1.5, 3.0, step=0.25)
    risk_per_trade    = trial.suggest_float("risk_per_trade",     0.02, 0.07, step=0.01)

    # Ensure target always larger than stop.
    if target_r <= stop_r:
        target_r = stop_r + 0.25

    sparm = {
        "mom_vwap_z_min":    mom_vwap_z_min,
        "mom_vol_ratio_min": mom_vol_ratio_min,
        "mom_delta_min":     mom_delta_min,
        "rev_vwap_z_max":    rev_vwap_z_max,
        "rev_vol_spike_min": rev_vol_spike_min,
        "stop_r":            stop_r,
        "target_r":          target_r,
        "risk_per_trade":    risk_per_trade,
    }

    score, period_results = run_trial_across_periods(sparm)

    # ── Attach per-period stats as Optuna user attributes for later analysis ──
    trial.set_user_attr("period_results", json.dumps([
        {k: v for k, v in r.items() if k != "period"} for r in period_results
    ]))
    trial.set_user_attr("total_trades_all", sum(r["total_trades"] for r in period_results))
    trial.set_user_attr("effective_target_r", target_r)

    return score


def write_csv_row(trial: optuna.Trial, score: float, period_results: list[dict]) -> None:
    """Append one result row to the CSV file."""
    sparm = trial.params

    totals = {
        "trades": sum(r["total_trades"] for r in period_results),
        "pf":     sum(r["profit_factor"] for r in period_results) / max(len(period_results), 1),
        "wr":     sum(r["win_rate"] for r in period_results)      / max(len(period_results), 1),
        "dd":     sum(r["max_drawdown"] for r in period_results)  / max(len(period_results), 1),
        "pnl":    sum(r["avg_pnl"] for r in period_results)       / max(len(period_results), 1),
        "ret":    sum(r.get("total_return", 0.0) for r in period_results) / max(len(period_results), 1),
    }

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trial.number,
        round(score, 4),
        round(totals["pf"], 3),
        round(totals["wr"], 2),
        round(totals["dd"], 2),
        round(totals["pnl"], 4),
        round(totals["ret"], 3),
        totals["trades"],
        sparm.get("mom_vwap_z_min"), sparm.get("mom_vol_ratio_min"), sparm.get("mom_delta_min"),
        sparm.get("rev_vwap_z_max"), sparm.get("rev_vol_spike_min"),
        sparm.get("stop_r"),
        trial.user_attrs.get("effective_target_r", sparm.get("target_r")),
        sparm.get("risk_per_trade"),
    ]

    # Serialise file access — prevents interleaved rows or double-header when n_jobs > 1
    with _csv_lock:
        file_exists = os.path.isfile(RESULTS_CSV)  # re-check under lock
        with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADER)
            writer.writerow(row)


def print_leaderboard(study: optuna.Study, top_n: int = 10) -> None:
    """Print the top-N trials so far."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not trials:
        print("  (no completed trials yet)")
        return
    trials.sort(key=lambda t: t.value, reverse=True)
    print(f"\n" + "-" * 72)
    print(f"  TOP {min(top_n, len(trials))} CONFIGURATIONS (of {len(trials)} total trials):")
    print("-" * 72)
    header = f"{'Rank':<5} {'Score':>7} {'PF':>6} {'Trades':>7}  {'Params Summary'}"
    print(header)
    for rank, t in enumerate(trials[:top_n], 1):
        p = t.params
        pts_json = t.user_attrs.get("period_results", "[]")
        pts = json.loads(pts_json)
        avg_pf = sum(r.get("profit_factor", 0) for r in pts) / max(len(pts), 1)
        total_trades = t.user_attrs.get("total_trades_all", "?")
        summary = (
            f"MOM:Z>{p['mom_vwap_z_min']:.1f}/VR>{p['mom_vol_ratio_min']:.1f}/DR>{p['mom_delta_min']:.2f} "
            f"REV:Z<{p['rev_vwap_z_max']:.1f}/VS>{p['rev_vol_spike_min']:.1f} "
            f"RR:{p['stop_r']}/{p['target_r']} Risk:{p['risk_per_trade']:.0%}"
        )
        print(f"  {rank:<3} {t.value:>7.4f} {avg_pf:>6.2f} {str(total_trades):>7}  {summary}")
    print("-" * 72 + "\n")


def export_best_params(study: optuna.Study) -> None:
    """Write the best trial's params to test_params.txt (strategy params) and test_params_full.json (all params)."""
    best = study.best_trial
    p = best.params
    # Use the clamped effective_target_r stored as a user attribute during objective().
    # trial.params["target_r"] is the raw Optuna suggestion (may be ≤ stop_r); the
    # actual backtest used the clamped value stored in effective_target_r.
    effective_target_r = best.user_attrs.get("effective_target_r", p["target_r"])

    # ── 1. Strategy param file (read by load_strategy_params) ───────────────
    header = "mom_vwap_z_min,mom_vol_ratio_min,mom_delta_min,rev_vwap_z_max,rev_vol_spike_min,stop_r,target_r,risk_per_trade"
    values = (
        f"Input: {p['mom_vwap_z_min']:.1f},{p['mom_vol_ratio_min']:.1f},{p['mom_delta_min']:.2f},"
        f"{p['rev_vwap_z_max']:.1f},{p['rev_vol_spike_min']:.1f},"
        f"{p['stop_r']:.2f},{effective_target_r:.2f},{p['risk_per_trade']:.2f}"
    )
    with open(BEST_PARAMS_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(values + "\n")

    # ── 2. Full JSON file with ALL optimized params ──────────────────────────
    full_params_path = os.path.join(THIS_DIR, "test_params_full.json")
    full_params = {
        "mom_vwap_z_min":    p["mom_vwap_z_min"],
        "mom_vol_ratio_min": p["mom_vol_ratio_min"],
        "mom_delta_min":     p["mom_delta_min"],
        "rev_vwap_z_max":    p["rev_vwap_z_max"],
        "rev_vol_spike_min": p["rev_vol_spike_min"],
        "stop_r":            p["stop_r"],
        "target_r":          effective_target_r,
        "risk_per_trade":    p["risk_per_trade"],
        "_composite_score":  best.value,
        "_trial_number":     best.number,
    }
    with open(full_params_path, "w", encoding="utf-8") as f:
        json.dump(full_params, f, indent=2)

    # ── 3. Print the full backtest CLI command ────────────────────────────────
    print(f"\n[OK] Best params exported:")
    print(f"     test_params.txt         → strategy entry/exit params")
    print(f"     test_params_full.json   → all params including new ones")
    print(f"     Score: {best.value:.4f}")
    print(f"\n  Full backtest command:")
    print(f"    python backtest/run_backtest_v2.py \\")
    print(f"      --start 2023-01-01 --end 2023-12-31 \\")
    print(f"      --param-file backtest/test_params.txt \\")
    print(f"      --regime-dwell-days 3 \\")
    print(f"      --slippage 0.0003 \\")
    print(f"      --no-fetch")


def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Called after every completed trial. Handles logging and leaderboard."""
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    score = trial.value or 0.0
    p = trial.params
    pts_json = trial.user_attrs.get("period_results", "[]")
    pts = json.loads(pts_json)
    total_trades = trial.user_attrs.get("total_trades_all", 0)

    # Print period breakdown
    period_summaries = []
    for period, r in zip(EVAL_PERIODS, pts):
        t = r.get("total_trades", 0)
        pf = r.get("profit_factor", 0.0)
        wr = r.get("win_rate", 0.0)
        dd = r.get("max_drawdown", 0.0)
        period_summaries.append(
            f"    {period['name']:<40} T={t:<4} PF={pf:.2f} WR={wr:.1f}% DD={dd:.1f}%"
        )

    is_best = trial.value >= study.best_value if study.best_value is not None else True
    star = " << NEW BEST!" if is_best else ""

    print(f"\n[Trial #{trial.number}]  Score={score:.4f}{star}")

    if is_best:
        # Attach period names to results before sending
        named_pts = [dict(r, name=ep["name"]) for r, ep in zip(pts, EVAL_PERIODS)]
        send_optimizer_best(trial.number, score, p, named_pts, total_trades)
    print(f"  Params: MOM:Z>{p.get('mom_vwap_z_min','?')}/VR>{p.get('mom_vol_ratio_min','?')}/DR>{p.get('mom_delta_min','?')} "
          f"REV:Z<{p.get('rev_vwap_z_max','?')}/VS>{p.get('rev_vol_spike_min','?')} "
          f"RR:{p['stop_r']}/{p['target_r']} Risk:{p.get('risk_per_trade','?')}  TotalTrades={total_trades}")
    for s in period_summaries:
        print(s)

    # Write to CSV
    try:
        # Re-run to get full period_results for CSV
        write_csv_row(trial, score, pts)
    except Exception as exc:
        print(f"  [!] CSV write failed: {exc}")

    # Print leaderboard every 10 trials
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_complete % 10 == 0:
        print_leaderboard(study, top_n=5)


def main():
    parser = argparse.ArgumentParser(description="V2 Bayesian Parameter Optimizer")
    parser.add_argument("--reset",        action="store_true",  help="Delete existing study DB and results CSV, then start fresh")
    parser.add_argument("--resume",       action="store_true",  help="Resume from existing SQLite study")
    parser.add_argument("--n-jobs",       type=int, default=1,   help="Number of parallel Optuna workers")
    parser.add_argument("--show-best",    type=int, default=0,   help="Show top-N results and exit")
    parser.add_argument("--export-best",  action="store_true",  help="Export best params to file and exit")
    parser.add_argument("--n-trials",     type=int, default=0,   help="Stop after N trials (0 = infinite)")
    args = parser.parse_args()

    # -- Reset: wipe old study DB + CSV so the next run starts completely fresh --
    if args.reset:
        deleted = []
        for path in (STUDY_DB, RESULTS_CSV):
            if os.path.exists(path):
                os.remove(path)
                deleted.append(os.path.basename(path))
        if deleted:
            print(f"  [RESET] Deleted: {', '.join(deleted)}")
        else:
            print("  [RESET] Nothing to delete — already clean.")

    study_name  = "v2_backtest_optimizer"

    # -- Load or Create Study (load_if_exists=True handles both cases) ---------
    print("=" * 72)
    print("  V2 SMART PARAMETER OPTIMIZER  --  Bayesian Optimization (Optuna TPE)")
    print("=" * 72)

    # RDBStorage with a 30-second busy timeout prevents "database is locked"
    # crashes when n_jobs > 1 threads race to commit trial results.
    # WAL journal mode lets readers proceed concurrently with the single writer.
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{STUDY_DB}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    # Enable WAL mode on first connect so concurrent reads don't block writes
    with sqlite3.connect(STUDY_DB) as _con:
        _con.execute("PRAGMA journal_mode=WAL")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=len(WARM_START_SEEDS) + 5,
            multivariate=True,
            group=True,
        ),
        load_if_exists=True,
    )
    n_existing = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_existing > 0:
        print(f"  [OK] Resumed existing study: {n_existing} completed trials found.")
    else:
        print(f"  [OK] Created new study. Storage: {os.path.basename(STUDY_DB)}")

    # -- Show-only mode --------------------------------------------------------
    if args.show_best > 0:
        print_leaderboard(study, top_n=args.show_best)
        return

    if args.export_best:
        try:
            export_best_params(study)
        except Exception as e:
            print(f"  [!] No best trial found yet: {e}")
        return

    # -- Auto-fetch 5m bar data for all eval periods (skips if already cached) --
    _prefetch_eval_data()

    # -- Enqueue warm-start seeds (only for fresh studies) --------------------
    n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_done == 0:
        print(f"\n  Enqueueing {len(WARM_START_SEEDS)} warm-start seed configurations...")
        for seed in WARM_START_SEEDS:
            try:
                study.enqueue_trial(seed)
            except Exception:
                pass  # May already exist on resume
        print("  Warm seeds enqueued. These will run first.\n")

    print(f"\n  Evaluation periods: {len(EVAL_PERIODS)}")
    for p in EVAL_PERIODS:
        print(f"    * {p['name']}")
    print(f"\n  Scoring: PF*0.15 + WR*0.15 + AvgPnL*0.25 + TotalRet*0.30 + 1/DD*0.15")
    print(f"  Trade floor=40 (ramp below, +20% bonus above 100)")
    print(f"  Objective: 50% avg + 50% worst - 15% std  (all periods good > few extraordinary)")
    print(f"\n  Results CSV:  {os.path.basename(RESULTS_CSV)}")
    print(f"  Study DB:     {os.path.basename(STUDY_DB)}")
    print(f"\n  Press Ctrl+C to stop cleanly. Progress is auto-saved after every trial.")
    print("=" * 72 + "\n")

    # ── Main Optimization Loop ────────────────────────────────────────────────
    try:
        # Phase 1: drain warm-start seeds with a single worker.
        # SQLite has no row-level locking, so parallel workers racing for WAITING
        # trials causes "Cannot tell a COMPLETE trial" crashes.
        n_done_now = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        warm_remaining = max(0, len(WARM_START_SEEDS) - n_done_now)
        if warm_remaining > 0:
            print(f"  [Phase 1] Running {warm_remaining} warm-start seed(s) with n_jobs=1 ...")
            study.optimize(objective, n_trials=warm_remaining, n_jobs=1,
                           callbacks=[callback], catch=(Exception,), show_progress_bar=False)

        # Phase 2: main TPE exploration with requested parallelism.
        # No WAITING trials remain, so parallel workers only ever compete for
        # freshly-created trials — the SQLite race condition cannot occur.
        phase2 = (args.n_trials - len(WARM_START_SEEDS)) if args.n_trials > 0 else int(1e9)
        if phase2 > 0:
            if args.n_jobs > 1:
                print(f"  [Phase 2] TPE exploration with n_jobs={args.n_jobs} ...")
            study.optimize(objective, n_trials=phase2, n_jobs=args.n_jobs,
                           callbacks=[callback], catch=(Exception,), show_progress_bar=False)

    except KeyboardInterrupt:
        print("\n\n  Optimization interrupted by user. Saving state...")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_leaderboard(study, top_n=10)

    try:
        export_best_params(study)
    except Exception as e:
        print(f"  Could not export best params: {e}")

    # Sort CSV by composite score descending
    try:
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV)
            df = df.sort_values("composite_score", ascending=False)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"  CSV sorted by composite score: {os.path.basename(RESULTS_CSV)}")
    except Exception as e:
        print(f"  CSV sort warning: {e}")

    print("\n  Optimizer exited cleanly.")


if __name__ == "__main__":
    main()
