"""
pick_best.py — Export best parameters from v2_optimization_results.csv
=======================================================================
Reads the CSV written by the optimizer (never touches the SQLite DB),
finds the trial with the highest composite_score, writes:
  • test_params.txt         — legacy format read by load_strategy_params()
  • test_params_full.json   — all params including new ones (CLI flags)

Safe to run while the optimizer is still running.

Usage:
    python backtest/pick_best.py
    python backtest/pick_best.py --csv backtest/v2_optimization_results.csv
"""

import argparse
import json
import os
import sys

import pandas as pd

THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV    = os.path.join(THIS_DIR, "v2_optimization_results.csv")
PARAMS_TXT_OUT = os.path.join(THIS_DIR, "test_params.txt")
PARAMS_JSON_OUT= os.path.join(THIS_DIR, "test_params_full.json")


def pick_best(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Drop zero-score rows (no trades / penalised trials)
    valid = df[df["composite_score"] > 0]
    if valid.empty:
        print("[ERROR] No valid (non-zero score) trials in CSV yet.")
        sys.exit(1)

    best = valid.loc[valid["composite_score"].idxmax()]
    score = best["composite_score"]
    trial = int(best["trial_number"])

    # Apply the same clamp that objective() uses: if target_r <= stop_r (can happen
    # for trials where the CSV stored the raw Optuna suggestion before the fix, or
    # from older runs), use stop_r + 0.5 — the actual value the backtest ran with.
    stop_r_val   = float(best["stop_r"])
    target_r_val = float(best["target_r"])
    if target_r_val <= stop_r_val:
        target_r_val = stop_r_val + 0.5

    print(f"\n  Best trial : #{trial}")
    print(f"  Score      : {score:.4f}")
    print(f"  Trades     : {int(best['total_trades'])}")
    print(f"  PF         : {best['avg_profit_factor']:.3f}  |  "
          f"WR: {best['avg_win_rate']:.1f}%  |  "
          f"DD: {best['avg_drawdown']:.2f}%\n")

    # ── 1. test_params.txt  (legacy format read by load_strategy_params) ──────
    # long_bounce_pct / dip_bounce_pct were removed from the optimizer;
    # keep them as 0.0 so the parser doesn't break on the old key list.
    header = (
        "long_hybrid,long_exec,long_vwap,"
        "dip_hybrid,dip_exec,dip_vwap,"
        "long_delta_min,dip_delta_max,"
        "long_bounce_pct,dip_bounce_pct,"
        "stop_r,target_r,risk_per_trade"
    )
    values = (
        f"Input: "
        f"{best['long_hybrid']:.1f},{best['long_exec']:.1f},{best['long_vwap']:.1f},"
        f"{best['dip_hybrid']:.1f},{best['dip_exec']:.1f},{best['dip_vwap']:.1f},"
        f"{best['long_delta_min']:.2f},{best['dip_delta_max']:.2f},"
        f"0.0,0.0,"
        f"{stop_r_val:.1f},{target_r_val:.1f},{best['risk_per_trade']:.2f}"
    )

    with open(PARAMS_TXT_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"  [OK] Written: {os.path.basename(PARAMS_TXT_OUT)}")

    # ── 2. test_params_full.json  (all params, CLI-flag values) ──────────────
    # loss_cooldown_threshold is stored as positive magnitude in the CSV;
    # run_backtest_v2.py expects it as a negative number in the JSON.
    max_hold   = int(best["max_hold_trading_days"])
    cooldown   = int(best["ticker_cooldown_days"])
    loss_thr   = float(best["loss_cooldown_threshold"])   # positive magnitude
    ext_cd     = int(best["ticker_cooldown_extended_days"])

    full_params = {
        "long_hybrid":                   float(best["long_hybrid"]),
        "long_exec":                     float(best["long_exec"]),
        "long_vwap":                     float(best["long_vwap"]),
        "long_delta_min":                float(best["long_delta_min"]),
        "dip_hybrid":                    float(best["dip_hybrid"]),
        "dip_exec":                      float(best["dip_exec"]),
        "dip_vwap":                      float(best["dip_vwap"]),
        "dip_delta_max":                 float(best["dip_delta_max"]),
        "stop_r":                        stop_r_val,
        "target_r":                      target_r_val,
        "dip_stop_r":                    float(best.get("dip_stop_r", 2.0)),
        "dip_target_r":                  float(best.get("dip_target_r", 2.0)),
        "risk_per_trade":                float(best["risk_per_trade"]),
        "max_hold_trading_days":         max_hold,
        "ticker_cooldown_days":          cooldown,
        "loss_cooldown_threshold":       -loss_thr,   # stored negative
        "ticker_cooldown_extended_days": ext_cd,
        "_composite_score":              float(score),
        "_trial_number":                 trial,
    }

    with open(PARAMS_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(full_params, f, indent=2)

    print(f"  [OK] Written: {os.path.basename(PARAMS_JSON_OUT)}")

    # ── 3. Print the backtest command ─────────────────────────────────────────
    print(f"\n  Run backtest with these params:")
    print(f"    python backtest/run_backtest_v2.py \\")
    print(f"      --start 2023-01-01 --end 2023-12-31 \\")
    print(f"      --param-file backtest/test_params.txt \\")
    print(f"      --max-hold-days {max_hold} \\")
    print(f"      --cooldown-days {cooldown} \\")
    print(f"      --loss-cooldown-threshold -{loss_thr:.2f} \\")
    print(f"      --extended-cooldown-days {ext_cd} \\")
    print(f"      --regime-dwell-days 3 \\")
    print(f"      --slippage 0.0003 \\")
    print(f"      --no-fetch\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export best params from v2_optimization_results.csv to test_params.txt / test_params_full.json"
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help="Path to the results CSV (default: backtest/v2_optimization_results.csv)"
    )
    args = parser.parse_args()
    pick_best(args.csv)


if __name__ == "__main__":
    main()
