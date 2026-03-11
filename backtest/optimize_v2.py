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
import argparse
import logging
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

# Also silence Optuna's own verbose logging (we print our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
STUDY_DB   = os.path.join(THIS_DIR, "optimize_v2_study.db")
RESULTS_CSV = os.path.join(THIS_DIR, "v2_optimization_results.csv")
BEST_PARAMS_OUT = os.path.join(THIS_DIR, "test_params_optimized.txt")

# ── Multi-period evaluation windows ──────────────────────────────────────────
# Strategies must perform well across ALL regimes to score high.
EVAL_PERIODS = [
    {"name": "Jan-Mar 2023  (Bull Recovery)",    "start": "2023-01-01", "end": "2023-03-31"},
    {"name": "Aug-Oct 2023  (Distribution/Bear)", "start": "2023-08-01", "end": "2023-10-31"},
    {"name": "Jan-Feb 2024  (Momentum Bull)",     "start": "2024-01-01", "end": "2024-02-29"},
]

# ── Warm-start seeds (enqueue these as the first Optuna trials) ──────────────
WARM_START_SEEDS = [
    # Current best (test_params_optimized.txt values)
    {
        "long_hybrid": 65.0, "long_exec": 40.0, "long_vwap": 2.5,
        "short_hybrid": 65.0, "short_exec": 40.0, "short_vwap": -2.5,
        "long_delta_min": 0.60, "short_delta_max": 0.40,
        "stop_r": 3.0, "target_r": 3.5,
    },
    # Original baseline (test_params.txt values)
    {
        "long_hybrid": 56.0, "long_exec": 33.0, "long_vwap": 2.5,
        "short_hybrid": 60.0, "short_exec": 39.0, "short_vwap": -2.5,
        "long_delta_min": 0.55, "short_delta_max": 0.43,
        "stop_r": 2.0, "target_r": 3.5,
    },
    # Tighter delta, bigger stops — high conviction only
    {
        "long_hybrid": 70.0, "long_exec": 45.0, "long_vwap": 2.0,
        "short_hybrid": 70.0, "short_exec": 45.0, "short_vwap": -2.0,
        "long_delta_min": 0.65, "short_delta_max": 0.35,
        "stop_r": 3.5, "target_r": 4.0,
    },
    # Looser entry, tighter stop — breadth play
    {
        "long_hybrid": 60.0, "long_exec": 36.0, "long_vwap": 3.0,
        "short_hybrid": 60.0, "short_exec": 36.0, "short_vwap": -3.0,
        "long_delta_min": 0.55, "short_delta_max": 0.45,
        "stop_r": 2.5, "target_r": 3.0,
    },
]

# ── CSV Header ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "trial_number", "composite_score",
    "avg_profit_factor", "avg_win_rate", "avg_drawdown", "avg_pnl", "total_trades",
    "min_period_score",
    "long_hybrid", "long_exec", "long_vwap",
    "short_hybrid", "short_exec", "short_vwap",
    "long_delta_min", "short_delta_max",
    "stop_r", "target_r",
]


def compute_composite_score(results: list[dict]) -> float:
    """
    Compute the composite score across all evaluation periods.

    Components (each period):
      - Profit Factor × 0.40
      - Win Rate (0-1)  × 0.30
      - 1/max_drawdown  × 0.15   (rewards staying small)
      - avg_pnl         × 0.15

    Hard penalties:
      - total_trades < 12 across all periods → ×0.4   (prevent luck-based overfitting)
      - any period drawdown > 35%             → ×0.2   (blowup risk)
      - profit_factor < 0.8 in any period     → ×0.3   (must not hemorrhage in any regime)

    The final score returned is the MINIMUM period score — the strategy must
    survive EVERY regime, not just thrive in one.
    """
    if not results:
        return 0.0

    period_scores = []
    total_trades_all = sum(r.get("total_trades", 0) for r in results)

    for r in results:
        pf  = r.get("profit_factor", 0.0)
        wr  = r.get("win_rate", 0.0) / 100.0
        dd  = abs(r.get("max_drawdown", 50.0))
        pnl = r.get("avg_pnl", 0.0)
        trades = r.get("total_trades", 0)

        # Cap PF to prevent inf from blowing up the score (0 losses = inf PF)
        pf_capped = 5.0 if pf == float('inf') else min(pf, 5.0)

        # Base score
        dd_component = 1.0 / max(dd, 1.0)
        score = (pf_capped * 0.40) + (wr * 0.30) + (dd_component * 0.15) + (pnl * 0.15)

        # Hard penalties per period
        if trades < 10:                 score *= 0.2  # Severe penalty for tiny sample size
        elif trades < 25:               score *= 0.5  # Heavy penalty for low trades
        
        if dd > 35.0:                   score *= 0.2
        if pf < 0.8:                    score *= 0.3

        period_scores.append(score)

    # Global check: we want at least 90 trades across all 3 periods combined for statistical significance
    if total_trades_all < 90:
        factor = max(0.01, (total_trades_all / 90.0) ** 2) # Quadratic penalty to aggressively push trades up
        period_scores = [s * factor for s in period_scores]

    # Return the minimum period score to force cross-regime robustness
    return min(period_scores)


def run_trial_across_periods(sparm: dict, verbose: bool = False) -> tuple[float, list[dict]]:
    """Run the backtest across all 3 evaluation periods and return (composite_score, per_period_results)."""
    all_results = []
    for period in EVAL_PERIODS:
        params = {
            "stop_r":           sparm["stop_r"],
            "target_r":         sparm["target_r"],
            "risk_per_trade":   0.01,           # Volatility Parity baseline
            "max_positions":    5,
            "max_hold_days":    0,              # Strictly intraday
        }

        try:
            result = run_backtest_v2(
                start_date=period["start"],
                end_date=period["end"],
                top_n=10,
                fetch_5m=True,
                verbose=False,
                always_long=False,
                params=params,
                sparm=sparm,
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
        })

    composite = compute_composite_score(all_results)
    return composite, all_results


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest parameters, run backtests, return composite score."""

    # ── Parameter Search Space ───────────────────────────────────────────────
    # Long (Bull) side
    long_hybrid   = trial.suggest_float("long_hybrid",   55.0, 80.0, step=1.0)
    long_exec     = trial.suggest_float("long_exec",     30.0, 55.0, step=1.0)
    long_vwap     = trial.suggest_float("long_vwap",      0.0,  4.0, step=0.5)
    long_delta    = trial.suggest_float("long_delta_min", 0.50, 0.75, step=0.01)

    # Short (Bear) side  — symmetric structure
    short_hybrid  = trial.suggest_float("short_hybrid",  55.0, 80.0, step=1.0)
    short_exec    = trial.suggest_float("short_exec",    30.0, 55.0, step=1.0)
    short_vwap    = trial.suggest_float("short_vwap",   -4.0,  0.0,  step=0.5)
    short_delta   = trial.suggest_float("short_delta_max", 0.25, 0.50, step=0.01)

    # Risk / exit
    stop_r        = trial.suggest_float("stop_r",   1.5, 4.0, step=0.5)
    target_r      = trial.suggest_float("target_r", 2.0, 5.0, step=0.5)

    # Ensure target always larger than stop
    if target_r <= stop_r:
        return 0.0

    sparm = {
        "long_hybrid":     long_hybrid,
        "long_exec":       long_exec,
        "long_vwap":       long_vwap,
        "long_delta_min":  long_delta,
        "long_bounce_pct": 0.0,
        "short_hybrid":    short_hybrid,
        "short_exec":      short_exec,
        "short_vwap":      short_vwap,
        "short_delta_max": short_delta,
        "short_bounce_pct": 0.0,
        "stop_r":          stop_r,
        "target_r":        target_r,
    }

    score, period_results = run_trial_across_periods(sparm)

    # ── Attach per-period stats as Optuna user attributes for later analysis ──
    trial.set_user_attr("period_results", json.dumps([
        {k: v for k, v in r.items() if k != "period"} for r in period_results
    ]))
    trial.set_user_attr("total_trades_all", sum(r["total_trades"] for r in period_results))

    return score


def write_csv_row(trial: optuna.Trial, score: float, period_results: list[dict]) -> None:
    """Append one result row to the CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)
    sparm = trial.params

    totals = {
        "trades": sum(r["total_trades"] for r in period_results),
        "pf":     sum(r["profit_factor"] for r in period_results) / max(len(period_results), 1),
        "wr":     sum(r["win_rate"] for r in period_results)      / max(len(period_results), 1),
        "dd":     sum(r["max_drawdown"] for r in period_results)  / max(len(period_results), 1),
        "pnl":    sum(r["avg_pnl"] for r in period_results)       / max(len(period_results), 1),
    }

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trial.number,
        round(score, 4),
        round(totals["pf"], 3),
        round(totals["wr"], 2),
        round(totals["dd"], 2),
        round(totals["pnl"], 4),
        totals["trades"],
        round(score, 4),
        sparm.get("long_hybrid"),  sparm.get("long_exec"),  sparm.get("long_vwap"),
        sparm.get("short_hybrid"), sparm.get("short_exec"), sparm.get("short_vwap"),
        sparm.get("long_delta_min"), sparm.get("short_delta_max"),
        sparm.get("stop_r"), sparm.get("target_r"),
    ]

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
            f"L:{p['long_hybrid']:.0f}/{p['long_exec']:.0f}/D{p['long_delta_min']:.2f}/Z{p['long_vwap']:.1f} "
            f"S:{p['short_hybrid']:.0f}/{p['short_exec']:.0f}/D{p['short_delta_max']:.2f}/Z{p['short_vwap']:.1f} "
            f"RR:{p['stop_r']}/{p['target_r']}"
        )
        print(f"  {rank:<3} {t.value:>7.4f} {avg_pf:>6.2f} {str(total_trades):>7}  {summary}")
    print("-" * 72 + "\n")


def export_best_params(study: optuna.Study) -> None:
    """Write the best trial's params to test_params_optimized.txt."""
    best = study.best_trial
    p = best.params
    header = "long_hybrid,long_exec,long_vwap,short_hybrid,short_exec,short_vwap,long_delta_min,short_delta_max,long_bounce_pct,short_bounce_pct,stop_r,target_r"
    values = (
        f"Input: {p['long_hybrid']},{p['long_exec']},{p['long_vwap']},"
        f"{p['short_hybrid']},{p['short_exec']},{p['short_vwap']},"
        f"{p['long_delta_min']},{p['short_delta_max']},"
        f"0.0,0.0,"
        f"{p['stop_r']},{p['target_r']}"
    )
    with open(BEST_PARAMS_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(values + "\n")
    print(f"\n[OK] Best params exported to: {os.path.basename(BEST_PARAMS_OUT)}")
    print(f"     Score: {best.value:.4f} | Params: {values}")


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
    print(f"  Params: L:{p['long_hybrid']:.0f}/{p['long_exec']:.0f}/D{p['long_delta_min']:.2f}/Z{p['long_vwap']:.1f} "
          f"S:{p['short_hybrid']:.0f}/{p['short_exec']:.0f}/D{p['short_delta_max']:.2f}/Z{p['short_vwap']:.1f} "
          f"RR:{p['stop_r']}/{p['target_r']}  TotalTrades={total_trades}")
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
    parser.add_argument("--resume",       action="store_true",  help="Resume from existing SQLite study")
    parser.add_argument("--n-jobs",       type=int, default=1,   help="Number of parallel Optuna workers")
    parser.add_argument("--show-best",    type=int, default=0,   help="Show top-N results and exit")
    parser.add_argument("--export-best",  action="store_true",  help="Export best params to file and exit")
    parser.add_argument("--n-trials",     type=int, default=0,   help="Stop after N trials (0 = infinite)")
    args = parser.parse_args()

    storage_url = f"sqlite:///{STUDY_DB}"
    study_name  = "v2_backtest_optimizer"

    # -- Load or Create Study (load_if_exists=True handles both cases) ---------
    print("=" * 72)
    print("  V2 SMART PARAMETER OPTIMIZER  --  Bayesian Optimization (Optuna TPE)")
    print("=" * 72)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
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
    print(f"\n  Scoring: Composite (PF*0.40 + WR*0.30 + 1/DD*0.15 + AvgPnL*0.15)")
    print(f"  Objective: Minimize worst-case period score (force cross-regime survival)")
    print(f"\n  Results CSV:  {os.path.basename(RESULTS_CSV)}")
    print(f"  Study DB:     {os.path.basename(STUDY_DB)}")
    print(f"\n  Press Ctrl+C to stop cleanly. Progress is auto-saved after every trial.")
    print("=" * 72 + "\n")

    # ── Main Optimization Loop ────────────────────────────────────────────────
    n_trials = args.n_trials if args.n_trials > 0 else int(1e9)
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=args.n_jobs,
            callbacks=[callback],
            catch=(Exception,),
            show_progress_bar=False,
        )
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
