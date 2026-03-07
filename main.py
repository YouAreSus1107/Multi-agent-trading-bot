"""
War-Room Bot -- Main Entry Point (Day Trading Architecture)
Auto-detects market hours. Runs assessment at market close.
Research team preps when market is closed.

Usage:
    python main.py              # Auto mode (trading when open, research when closed)
    python main.py --force      # Force trading cycle regardless of hours
    python main.py --once       # Single cycle and exit
    python main.py --assess     # Run end-of-day assessment now
"""

# Force UTF-8 on Windows
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from graph.workflow import build_workflow, risk_agent, broker_service, research_team, news_analyst
from agents.assessment_agent import AssessmentAgent, REPORTS_DIR
from dashboard.terminal_ui import render_dashboard, console
from services.discord_notifier import send_discord_update, send_startup_message, send_shutdown_message, send_research_update
from config import (
    FAST_LOOP_SECONDS, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR,
    DAY_TRADE_TICKERS, PORTFOLIO_ALLOCATION,
)
import os
from utils.logger import get_logger
from rich.panel import Panel
from rich import box

logger = get_logger("main")

assessment_agent = AssessmentAgent()

# Build the day-trade ticker set (everything except mega-caps)
_MEGA_CAP_SET = set(PORTFOLIO_ALLOCATION.get("mega_cap_tickers", []))
_ALL_DAY_TRADE_TICKERS: set = set()
for _group in DAY_TRADE_TICKERS.values():
    _ALL_DAY_TRADE_TICKERS.update(_group.get("tickers", []))
_DAY_TRADE_ONLY = _ALL_DAY_TRADE_TICKERS - _MEGA_CAP_SET  # positions to close at EOD

# ─── Market time utilities ────────────────────────────────────────────────────

NY_TZ = ZoneInfo("America/New_York")

# 3 minutes before official close — hard deadline to submit the last cycle
EOD_CUTOFF_HOUR = 15
EOD_CUTOFF_MINUTE = 57

# 20 minutes before market open — pre-market briefing window
PREMARKET_BRIEFING_HOUR = 9
PREMARKET_BRIEFING_MINUTE = 10


def _now_ny() -> datetime:
    return datetime.now(NY_TZ)


def get_market_status() -> dict:
    """
    Auto-detect US market status (handles EST/EDT daylight savings).
    Returns: {open: bool, status: str, until: str, minutes_to_close: int}
    """
    now_ny = _now_ny()
    is_weekend = now_ny.weekday() >= 5
    market_open = now_ny.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = now_ny.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
    eod_cutoff = now_ny.replace(hour=EOD_CUTOFF_HOUR, minute=EOD_CUTOFF_MINUTE, second=0, microsecond=0)

    if is_weekend:
        return {"open": False, "status": "CLOSED (weekend)", "until": "Monday 9:30 AM ET", "minutes_to_close": 0}

    if now_ny < market_open:
        minutes_until = int((market_open - now_ny).total_seconds() / 60)
        return {"open": False, "status": "PRE-MARKET", "until": f"{minutes_until}m until open", "minutes_to_close": 0}

    if now_ny >= market_close:
        return {"open": False, "status": "CLOSED", "until": "Tomorrow 9:30 AM ET", "minutes_to_close": 0}

    # Within hours but past EOD cutoff (3:57pm) — treat as closed for trading
    if now_ny >= eod_cutoff:
        return {"open": False, "status": "EOD CUTOFF", "until": "Market closing", "minutes_to_close": 0}

    minutes_left = int((eod_cutoff - now_ny).total_seconds() / 60)
    return {"open": True, "status": "OPEN", "until": f"{minutes_left}m until close cutoff", "minutes_to_close": minutes_left}


def is_premarket_briefing_window() -> bool:
    """True if we're in the 9:10–9:30 AM ET window for pre-market briefing."""
    now_ny = _now_ny()
    if now_ny.weekday() >= 5:
        return False
    briefing_start = now_ny.replace(hour=PREMARKET_BRIEFING_HOUR, minute=PREMARKET_BRIEFING_MINUTE, second=0, microsecond=0)
    briefing_end = now_ny.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    return briefing_start <= now_ny < briefing_end


# ─── Cycle execution ──────────────────────────────────────────────────────────

def run_cycle(workflow, cycle_number: int) -> dict:
    """Run a single trading cycle."""
    logger.info(f"=== CYCLE {cycle_number} START ===")

    initial_state = {
        "technical_data": {},
        "fundamentals_data": {},
        "news_data": {},
        "sentiment_data": {},
        "research_decisions": [],
        "market_stance": "neutral",
        "overall_strategy": "",
        "research_notes": "",
        "trader_signals": [],
        "trader_strategy": "",
        "approved_trades": [],
        "rejected_by_risk": [],
        "halt": False,
        "hedge_mode": False,
        "hedge_actions": [],
        "risk_summary": "",
        "executed_trades": [],
        "execution_errors": [],
        "exit_signals": [],
        "portfolio": {},
        "positions": [],
        "vix_level": -1,
        "cycle_number": cycle_number,
        "cycle_timestamp": "",
        "candidate_tickers": [],
        "news_items": [],
    }

    try:
        result = workflow.invoke(initial_state)
        logger.info(f"=== CYCLE {cycle_number} COMPLETE ===")
        return result
    except Exception as e:
        logger.error(f"Cycle {cycle_number} failed: {e}")
        return {**initial_state, "risk_summary": f"CYCLE FAILED: {e}"}


# ─── EOD sequence ──────────────────────────────────────────────────────────────

def run_eod_sequence(cycle: int):
    """
    Full end-of-day sequence:
    1. Cancel all open orders
    2. Close all day-trade positions (keep mega-cap holds)
    3. Wait for fills to settle
    4. Run assessment and write report
    """
    logger.info("=" * 60)
    logger.info("⏰ MARKET CLOSE — Starting EOD sequence")
    logger.info("=" * 60)

    console.print(Panel(
        "[bold yellow]⏰ MARKET CLOSE DETECTED[/bold yellow]\n"
        "Cancelling open orders → Closing day-trade positions → Writing report",
        border_style="yellow",
        box=box.DOUBLE_EDGE,
        expand=False,
    ))

    # Step 1: Cancel all pending/open orders
    logger.info("[EOD] Step 1: Cancelling all open orders...")
    broker_service.cancel_all_orders()
    time.sleep(2)

    # Step 2: Close day-trade positions only (not mega-cap holds)
    logger.info(f"[EOD] Step 2: Closing day-trade positions: {sorted(_DAY_TRADE_ONLY)}")
    eod_closed = broker_service.close_day_trade_positions(_DAY_TRADE_ONLY)

    if eod_closed:
        wins = sum(1 for r in eod_closed if r.get("realized_pl_pct", 0) > 0)
        losses = len(eod_closed) - wins
        total_pl = sum(r.get("realized_pl_dollars", 0) for r in eod_closed)
        console.print(Panel(
            f"[bold]EOD LIQUIDATION COMPLETE[/bold]\n"
            f"Closed {len(eod_closed)} day-trade position(s) | "
            f"[green]{wins} wins[/green] / [red]{losses} losses[/red]\n"
            f"Day-trade P&L: [{'green' if total_pl >= 0 else 'red'}]${total_pl:+,.0f}[/]",
            border_style="yellow",
            box=box.ROUNDED,
            expand=False,
        ))
    else:
        console.print("[dim]No day-trade positions to close[/dim]")

    # Step 3: Wait for fills to settle
    logger.info("[EOD] Step 3: Waiting 30s for fills to settle...")
    time.sleep(30)

    # Step 4: Run assessment
    run_assessment(cycle, eod_closed)


def run_assessment(cycle: int, eod_closed: list = None):
    """Run end-of-day assessment and write detailed report."""
    console.print("[bold yellow]📝 Running end-of-day assessment...[/bold yellow]")
    try:
        portfolio = broker_service.get_account()
        positions = broker_service.get_positions()
        result = assessment_agent.assess_day(
            portfolio, positions, eod_closed or [], cycle
        )

        lessons = result.get("lessons", [])
        mistakes = result.get("mistakes", [])
        grade = result.get("strategy_grade", "?")
        pnl = result.get("pnl_summary", {})

        console.print(Panel(
            f"[bold]END-OF-DAY REPORT[/bold]\n"
            f"Strategy: [bold]{grade}[/bold] | Timing: [bold]{result.get('timing_grade','?')}[/bold]\n"
            f"Day-Trade Tier: [bold]{result.get('day_trade_tier_grade','?')}[/bold] | "
            f"Mega-Cap Tier: [bold]{result.get('mega_cap_tier_grade','?')}[/bold]\n"
            f"P&L: [{'green' if pnl.get('total_pnl_pct',0) >= 0 else 'red'}]"
            f"{pnl.get('total_pnl_pct',0):+.2f}%[/] (${pnl.get('total_pnl_dollars',0):+,.0f})\n\n"
            f"[bold yellow]Mistakes:[/bold yellow]\n" +
            "\n".join(f"  ⚠ {m}" for m in mistakes[:3]) +
            f"\n\n[bold green]Lessons:[/bold green]\n" +
            "\n".join(f"  • {l}" for l in lessons[:4]),
            border_style="yellow",
            box=box.DOUBLE_EDGE,
            expand=False,
        ))
    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        console.print(f"[red]Assessment error: {e}[/red]")


# ─── Overnight research ───────────────────────────────────────────────────────

def run_market_closed_mode(cycle: int):
    """When market is closed: overnight research prep only."""
    console.print("[dim cyan]Market closed — running overnight research...[/dim cyan]")
    try:
        positions = broker_service.get_positions()
        news_data = news_analyst.analyze()
        time.sleep(2)
        prep = research_team.research_while_closed(news_data, positions, cycle)

        watch = prep.get("watch_list", [])
        risks = prep.get("overnight_risks", [])

        console.print(Panel(
            f"[bold cyan]OVERNIGHT RESEARCH[/bold cyan]\n"
            f"[green]Watch List:[/green] {', '.join(watch[:8]) if watch else 'None'}\n"
            f"[red]Risks:[/red] {', '.join(r[:60] for r in risks[:3]) if risks else 'None'}\n"
            f"[dim]{prep.get('prep_notes', '')[:200]}[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=False,
        ))
        send_research_update(prep)
    except Exception as e:
        logger.error(f"Market-closed research failed: {e}")
        console.print(f"[dim red]Research error: {e}[/dim red]")


def run_premarket_briefing():
    """Generate and display the pre-market briefing from yesterday's report."""
    console.print("[bold cyan]📋 Generating pre-market briefing from yesterday's report...[/bold cyan]")
    try:
        briefing = assessment_agent.generate_premarket_briefing()
        if not briefing:
            console.print("[dim]No previous report — skipping briefing[/dim]")
            return

        watch = [w.get("ticker", "") for w in briefing.get("watch_list", [])]
        avoid = [a.get("ticker", "") for a in briefing.get("avoid_list", [])]

        console.print(Panel(
            f"[bold cyan]PRE-MARKET BRIEFING[/bold cyan]\n\n"
            f"[bold]Key Lessons from Yesterday:[/bold]\n" +
            "\n".join(f"  📌 {l}" for l in briefing.get("key_lessons", [])[:3]) +
            f"\n\n[bold yellow]Rule of the Day:[/bold yellow] {briefing.get('rule_of_the_day', '')}\n"
            f"[green]Watch:[/green] {', '.join(watch)}\n"
            f"[red]Avoid:[/red] {', '.join(avoid)}\n"
            f"[dim]{briefing.get('market_context', '')}[/dim]\n"
            f"[bold]Opening Bias:[/bold] {briefing.get('opening_bias', 'neutral')}",
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
            expand=False,
        ))
    except Exception as e:
        logger.error(f"Pre-market briefing failed: {e}")
        console.print(f"[dim red]Briefing error: {e}[/dim red]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="War-Room Day Trading Bot")
    parser.add_argument("--force", action="store_true", help="Force trading regardless of market hours")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--assess", action="store_true", help="Run end-of-day assessment now")
    args = parser.parse_args()

    # Send Notification to server
    send_startup_message()

    console.print(Panel(
        "[bold red]=== WAR-ROOM DAY TRADING BOT ===[/bold red]\n"
        "[dim]Multi-Agent Day Trading System[/dim]\n"
        "[dim]Analysts: Technical + Fundamentals + News + Sentiment[/dim]\n"
        "[dim]Research: Bull/Bear Debate → Risk → Execute[/dim]\n"
        f"[dim]EOD cutoff: {EOD_CUTOFF_HOUR}:{EOD_CUTOFF_MINUTE:02d} ET | "
        f"Day-trade tickers: {len(_DAY_TRADE_ONLY)}[/dim]",
        border_style="red",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
        expand=False,
    ))

    # Load yesterday's lessons at startup
    lessons = assessment_agent.load_yesterdays_lessons()
    if lessons:
        console.print(Panel(
            "[bold]Yesterday's Lessons (from last report):[/bold]\n" +
            "\n".join(f"  • {l}" for l in lessons[:5]),
            border_style="yellow",
            box=box.ROUNDED,
            expand=False,
        ))

    # Run assessment if requested
    if args.assess:
        run_assessment(0)
        return

    # Build LangGraph workflow
    console.print("[cyan]Building LangGraph DAG...[/cyan]")
    workflow = build_workflow()
    console.print("[green]DAG compiled. All agents ready.[/green]")

    cycle = 0
    assessed_today = False
    premarket_briefed_today = False
    _last_market_was_open = False  # track open→close transition

    # Single cycle mode
    if args.once:
        console.print("[yellow]Running single cycle...[/yellow]\n")
        cycle += 1
        with console.status("[bold green]Executing analytical LLM research and quant modeling...", spinner="dots"):
            result = run_cycle(workflow, cycle)
            
        render_dashboard(result, cycle)
        send_discord_update(result, cycle)
        # Manually invoke shutdown for test cycles
        send_shutdown_message()
        return

    # Main loop
    market = get_market_status()
    console.print(f"[cyan]Market: {market['status']} ({market['until']})[/cyan]")
    console.print(f"[cyan]Loop interval: {FAST_LOOP_SECONDS}s[/cyan]\n")

    try:
        while True:
            market = get_market_status()
            now_ny = _now_ny()

            # ── MARKET IS OPEN ─────────────────────────────────────────────
            if market["open"] or args.force:
                assessed_today = False
                premarket_briefed_today = False  # reset for next day
                cycle += 1

                if cycle == 1:
                    risk_agent.reset_daily()

                # Warn when approaching EOD cutoff
                mins_left = market.get("minutes_to_close", 99)
                if 1 <= mins_left <= 5:
                    console.print(f"[bold yellow]⚠ {mins_left} minutes until EOD cutoff! Wrapping up...[/bold yellow]")

                console.print(f"[dim]Market: {market['status']} — {market['until']}[/dim]")
                with console.status(f"[bold green]Running Cycle {cycle} Pipeline (Quant + LLM Debate)...", spinner="dots"):
                    result = run_cycle(workflow, cycle)
                render_dashboard(result, cycle)
                send_discord_update(result, cycle)

                if result.get("halt", False):
                    console.print("\n[bold red]TRADING HALTED by Risk Manager[/bold red]")

                _last_market_was_open = True
                console.print(f"\n[dim]Next cycle in {FAST_LOOP_SECONDS}s... (Ctrl+C to stop)[/dim]")
                time.sleep(FAST_LOOP_SECONDS)

            # ── MARKET IS CLOSED / EOD CUTOFF ─────────────────────────────
            else:
                # Detect if EOD sequence should run
                needs_eod = False
                if _last_market_was_open and not assessed_today and cycle > 0:
                    needs_eod = True
                elif market["status"] in ["CLOSED", "EOD CUTOFF"] and now_ny.weekday() < 5 and now_ny.hour >= 15:
                    # Bot was restarted post-market close — check if report is missing
                    report_path = os.path.join(REPORTS_DIR, f"{now_ny.strftime('%Y-%m-%d')}.md")
                    if not os.path.exists(report_path) and not assessed_today:
                        needs_eod = True

                if needs_eod:
                    _last_market_was_open = False
                    run_eod_sequence(cycle)
                    assessed_today = True
                    # Immediately after EOD, run the overnight research once
                    run_market_closed_mode(cycle)

                # Pre-market briefing window (9:00–9:30 AM ET)
                if 9 <= now_ny.hour < 10 and now_ny.minute < 30 and not premarket_briefed_today and now_ny.weekday() < 5:
                    run_premarket_briefing()
                    premarket_briefed_today = True

                # Reset daily flags when a new market day opens
                if market["status"] == "OPEN":
                    assessed_today = False

                status_line = (
                    f"[dim]Market {market['status']} — {market['until']}. "
                    f"{now_ny.strftime('%H:%M')} ET (Ctrl+C to stop)[/dim]"
                )
                console.print(status_line)

                # Smart sleeping
                cycle += 1
                if market["status"] in ["CLOSED", "CLOSED (weekend)"]:
                    # Deep sleep if it's the weekend or after 5pm/before 8am
                    if now_ny.weekday() >= 5 or now_ny.hour >= 17 or now_ny.hour < 8:
                        run_market_closed_mode(cycle)
                        time.sleep(3600)  # Sleep for 1 hour
                    else:
                        time.sleep(300)   # Sleep 5 minutes closer to the active windows
                else:
                    time.sleep(120)       # Normal 2 min out-of-bounds sleep

    except KeyboardInterrupt:
        console.print("\n[yellow]War-Room Bot shutting down...[/yellow]")
        logger.info("Bot stopped by user")
        send_shutdown_message()


if __name__ == "__main__":
    main()
