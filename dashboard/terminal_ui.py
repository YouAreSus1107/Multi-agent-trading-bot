"""
War-Room Bot -- Rich Terminal Dashboard (Day Trading Architecture)
Uses vertical stacked layout to avoid chopped-off boxes on narrow terminals.
"""

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


console = Console()


def render_dashboard(state: dict, cycle: int = 0):
    """Render the full dashboard after each cycle."""
    console.clear()
    
    parts = []
    
    header = _render_header(cycle, state.get("cycle_timestamp", ""))
    if header: parts.append(header)
        
    strat = _render_strategy_bar(state)
    if strat: parts.append(strat)
        
    quant = _render_quant_table(state)
    if quant: parts.append(quant)
        
    research = _render_research_decisions(state)
    if research: parts.append(research)
        
    pos = _render_positions_table(state)
    if pos: parts.append(pos)
        
    sig = _render_signals_table(state)
    if sig: parts.append(sig)
        
    exec_panel = _render_execution_panel(state)
    if exec_panel: parts.append(exec_panel)
        
    risk = _render_risk_footer(state)
    if risk: parts.append(risk)

    console.print(Group(*parts))


def _render_header(cycle: int, timestamp: str):
    header_text = Text()
    header_text.append("<<  WAR-ROOM DAY TRADER  >>", style="bold red")
    header_text.append(f"\n  Cycle #{cycle}  |  {timestamp[:19]}  |  ", style="dim")
    header_text.append("ARMED", style="bold green")

    return Panel(
        header_text,
        border_style="red",
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    )


def _render_strategy_bar(state: dict):
    """One-line strategy + stance bar, then portfolio stats."""
    stance = state.get("market_stance", "unknown")
    strategy = state.get("overall_strategy", "")
    vix = state.get("vix_level", -1)
    portfolio = state.get("portfolio", {})
    sentiment = state.get("sentiment_data", {})
    news = state.get("news_data", {})

    stance_color = {"aggressive_bull": "green", "cautiously_bullish": "green",
                    "neutral": "yellow", "cautiously_bearish": "red",
                    "defensive": "red"}.get(stance, "white")

    mood = sentiment.get("market_mood", news.get("market_mood", "unknown"))
    niche = news.get("niche_tickers_discovered", [])
    vix_str = f"{vix:.1f}" if vix > 0 else "N/A"

    equity = portfolio.get("equity", 0)
    bp = portfolio.get("buying_power", 0)
    cash = portfolio.get("cash", 0)

    sents = sentiment.get("ticker_sentiments", {})
    sent_parts = []
    for t, d in list(sents.items())[:6]:
        if isinstance(d, dict):
            s = d.get("score", 0.5)
            c = "green" if s > 0.6 else ("red" if s < 0.4 else "yellow")
            sent_parts.append(f"[{c}]{t}:{s:.2f}[/]")

    info_lines = []
    info_lines.append(
        f"[{stance_color} bold]{stance.upper().replace('_', ' ')}[/]  |  "
        f"Mood: {mood}  |  VIX: {vix_str}"
    )
    info_lines.append(
        f"Equity: [bold green]${equity:,.0f}[/]  |  "
        f"BP: ${bp:,.0f}  |  Cash: ${cash:,.0f}"
    )
    if sent_parts:
        info_lines.append(f"Sentiment: {' | '.join(sent_parts)}")
    if niche:
        info_lines.append(f"[cyan]Discovered: {', '.join(niche[:8])}[/]")
    if strategy:
        info_lines.append(f"[dim]{strategy}[/]")

    events = news.get("events", [])
    if events:
        for ev in events[:3]:
            info_lines.append(f"  [yellow]*[/] {ev.get('headline', '')[:70]}")

    trader_strategy = state.get("trader_strategy", "")
    if trader_strategy:
        info_lines.append(f"\n[cyan]Trader Strat:[/] {trader_strategy}")

    return Panel(
        "\n".join(info_lines),
        title="[INTEL]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def _render_quant_table(state: dict):
    quant_dict = state.get("quant_data", {}).get("quant_data", {})
    if not quant_dict:
        return None

    table = Table(
        title="[bold magenta][QUANT ANALYSIS][/]",
        box=box.SIMPLE_HEAVY,
        border_style="magenta",
        expand=True,
        show_lines=False,
    )
    table.add_column("Ticker", style="bold", min_width=6)
    table.add_column("Alpha", justify="right")
    table.add_column("Beta", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("GBM Exp", justify="right")
    table.add_column("Prob Up", justify="right")

    # Only show up to 6 tickers to not overwhelm the UI
    for t, data in list(quant_dict.items())[:6]:
        factors = data.get("factors", {})
        gbm = data.get("gbm", {})
        xgboost = data.get("xgboost", {})
        
        alpha = factors.get("alpha", 0)
        alpha_color = "green" if alpha > 0 else "red"
        
        beta = factors.get("beta", 1)
        
        vol = gbm.get("volatility", 0)
        expected = gbm.get("expected_price", 0)
        
        prob = xgboost.get("prob_up", 0.5)
        prob_color = "green" if prob > 0.55 else ("red" if prob < 0.45 else "yellow")
        prob_str = f"[{prob_color}]{prob:.1%}[/]" if xgboost else "N/A"
        
        table.add_row(
            t,
            f"[{alpha_color}]{alpha:+.2f}[/]",
            f"{beta:.2f}",
            f"{vol:.1%}",
            f"${expected:.2f}",
            prob_str
        )
        
    return table


def _render_research_decisions(state: dict):
    decisions = state.get("research_decisions", [])
    if not decisions:
        return None

    table = Table(
        title="[bold cyan][LLM RESEARCH DEBATE][/]",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        expand=True,
        show_lines=False,
    )
    table.add_column("Ticker", style="bold", min_width=6)
    table.add_column("Action", justify="center", min_width=6)
    table.add_column("Conf", justify="right", min_width=5)
    table.add_column("Reasoning", ratio=1)

    for d in decisions[:6]:
        action = d.get("action", "").upper()
        if action == "BUY":
            a_col = "green"
        elif action in ["SELL", "AVOID"]:
            a_col = "red"
        else:
            a_col = "yellow"
            
        conf = d.get("conviction", 0)
        r = d.get("reasoning", "")
        # Truncate reasoning to fit nicely
        if len(r) > 85:
            r = r[:82] + "..."
            
        table.add_row(
            d.get("ticker", ""),
            f"[{a_col}]{action}[/]",
            f"{conf}%",
            f"[dim]{r}[/]"
        )
        
    return table


def _render_positions_table(state: dict):
    positions = state.get("positions", [])

    table = Table(
        title="[POSITIONS]",
        box=box.SIMPLE_HEAVY,
        border_style="blue",
        expand=True,
        show_lines=False,
    )
    table.add_column("Ticker", style="bold", min_width=6)
    table.add_column("Qty", justify="right", min_width=5)
    table.add_column("Entry", justify="right", min_width=8)
    table.add_column("Current", justify="right", min_width=8)
    table.add_column("P&L", justify="right", min_width=8)
    table.add_column("P&L %", justify="right", min_width=7)

    for pos in positions:
        pl = pos.get("unrealized_pl", 0)
        pl_pct = pos.get("unrealized_pl_pct", 0)
        pl_color = "green" if pl >= 0 else "red"
        table.add_row(
            pos.get("ticker", ""),
            str(int(pos.get("qty", 0))),
            f"${pos.get('entry_price', 0):,.2f}",
            f"${pos.get('current_price', 0):,.2f}",
            f"[{pl_color}]${pl:,.2f}[/]",
            f"[{pl_color}]{pl_pct:+.2f}%[/]",
        )

    if not positions:
        table.add_row("[dim]No positions[/]", "", "", "", "", "")

    return table


def _render_signals_table(state: dict):
    approved = state.get("approved_trades", [])
    rejected = state.get("rejected_by_risk", [])
    
    parts = []

    if approved:
        table = Table(
            title="[bold green][+] APPROVED TRADES[/]",
            box=box.SIMPLE_HEAVY,
            border_style="green",
            expand=True,
            show_lines=False,
        )
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Dir", justify="center", min_width=5)
        table.add_column("Qty", justify="right", min_width=5)
        table.add_column("$$$", justify="right", min_width=9)
        table.add_column("Sharpe", justify="center", min_width=6)
        table.add_column("Conf", justify="center", min_width=5)

        for trade in approved:
            direction = trade.get("direction", "").upper()
            dir_color = "green" if direction == "LONG" else "red"
            table.add_row(
                trade.get("ticker", ""),
                f"[{dir_color}]{direction}[/]",
                str(trade.get("qty", 0)),
                f"${trade.get('position_dollars', 0):,.0f}",
                f"{trade.get('sharpe_ratio', 0):.2f}",
                f"{trade.get('confidence', 0):.0%}" if trade.get('confidence') else "",
            )
        parts.append(table)

    if rejected:
        rej_text = " | ".join(
            f"[red]{r.get('ticker', '')}[/]: {r.get('reason', '')[:40]}"
            for r in rejected[:6]
        )
        parts.append(Text.from_markup(f"  [dim]Rejected:[/] {rej_text}"))
        
    if not parts:
        return None
    return Group(*parts)


def _render_execution_panel(state: dict):
    executed = state.get("executed_trades", [])
    errors = state.get("execution_errors", [])

    if not executed and not errors:
        return Text.from_markup("[dim]  No trades executed this cycle[/]")

    lines = []
    for t in executed:
        lines.append(
            f"[green][+][/] {t.get('side', '').upper()} {t.get('qty', '')} "
            f"{t.get('ticker', '')} — {t.get('status', '')}"
        )
    for e in errors:
        lines.append(
            f"[red][!][/] FAILED: {e.get('ticker', '')} — {e.get('error', '')[:50]}"
        )

    return Panel(
        "\n".join(lines),
        title="[EXECUTION]",
        border_style="magenta",
        box=box.ROUNDED,
    )


def _render_risk_footer(state: dict):
    risk_summary = state.get("risk_summary", "No risk data")
    halt = state.get("halt", False)
    hedge = state.get("hedge_mode", False)

    if halt:
        style = "bold white on red"
        prefix = "[HALT]"
    elif hedge:
        style = "bold black on yellow"
        prefix = "[HEDGE]"
    else:
        style = "green"
        prefix = "[OK]"

    return Panel(
        f"{prefix} {risk_summary}",
        title="[RISK]",
        border_style="bold red" if halt else "green",
        style=style,
        box=box.HEAVY,
    )
