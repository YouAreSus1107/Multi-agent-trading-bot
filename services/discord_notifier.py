import requests
import json
from datetime import datetime
from config import DISCORD_WEBHOOK_URL
from utils.logger import get_logger

logger = get_logger("discord_notifier")

def send_startup_message():
    """Sends a notification that the bot has started."""
    if not DISCORD_WEBHOOK_URL: return
    try:
        payload = {
            "username": "War-Room Day Trader",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
            "embeds": [{
                "title": "🟢 War-Room Bot Started",
                "color": 0x00FF00,
                "description": "The LangGraph Day Trading bot has been initialized and is now actively monitoring the market.",
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Startup discord notification failed: {e}")


def send_shutdown_message():
    """Sends a notification that the bot has shut down."""
    if not DISCORD_WEBHOOK_URL: return
    try:
        payload = {
            "username": "War-Room Day Trader",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
            "embeds": [{
                "title": "🔴 War-Room Bot Shut Down",
                "color": 0xFF0000,
                "description": "The bot loop has been manually terminated or stopped.",
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Shutdown discord notification failed: {e}")


def send_research_update(prep: dict):
    """Sends overnight/weekend research findings to Discord."""
    if not DISCORD_WEBHOOK_URL: return
    try:
        watch = prep.get("watch_list", [])
        risks = prep.get("overnight_risks", [])
        prep_notes = prep.get("prep_notes", "")
        
        desc = ""
        if watch:
            desc += f"**🟢 Watch List:** {', '.join(watch[:10])}\n\n"
        if risks:
            desc += f"**🔴 Key Risks:**\n" + "\n".join(f"• {r[:100]}" for r in risks[:3]) + "\n\n"
        if prep_notes:
            desc += f"**📝 Notes:** {prep_notes[:1000]}"
            
        if not desc:
            desc = "No actionable research generated this cycle."
            
        payload = {
            "username": "War-Room Day Trader",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
            "embeds": [{
                "title": "🌙 Offline Research Update",
                "color": 0x3498db,
                "description": desc,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Research discord notification failed: {e}")


def send_discord_update(state: dict, cycle: int):
    """
    Sends a rich embedded summary of the current cycle state to Discord.
    """
    if not DISCORD_WEBHOOK_URL:
        logger.debug("DISCORD_WEBHOOK_URL not set. Skipping Discord notification.")
        return

    try:
        embeds = _build_embeds(state, cycle)
        
        payload = {
            "username": "War-Room Day Trader",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",  # Generic robot/trader icon
            "embeds": embeds
        }

        response = requests.post(
            DISCORD_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code not in (200, 204):
            logger.error(f"Failed to send Discord webhook: {response.status_code} - {response.text}")
        else:
            logger.info("Discord notification sent successfully.")
            
    except Exception as e:
        logger.error(f"Exception while sending Discord notification: {e}")


def send_optimizer_best(trial_number: int, score: float, params: dict,
                        period_results: list, total_trades: int) -> None:
    """
    Fires whenever the optimizer finds a new best composite score.
    Sends a compact rich embed with the key params and per-period breakdown.
    """
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        p = params

        # ── Colour by score ──────────────────────────────────────────────
        if score >= 1.0:
            color = 0x00FF00      # bright green
        elif score >= 0.7:
            color = 0x2ECC71      # softer green
        elif score >= 0.5:
            color = 0xF1C40F      # yellow
        else:
            color = 0xE67E22      # orange

        # ── Period breakdown table ───────────────────────────────────────
        rows = []
        for r in period_results:
            name  = r.get("name", "?")[:22]
            t     = r.get("total_trades", 0)
            pf    = r.get("profit_factor", 0.0)
            wr    = r.get("win_rate", 0.0)
            dd    = r.get("max_drawdown", 0.0)
            ret   = r.get("total_return", 0.0)
            rows.append(f"`{name:<22}` T={t:<3} PF={pf:.2f} WR={wr:.0f}% DD={dd:.1f}% Ret={ret:+.1f}%")
        periods_text = "\n".join(rows) if rows else "—"

        rr_ratio = round(p.get("target_r", 0) / max(p.get("stop_r", 1), 0.01), 2)

        embed = {
            "title": f"🏆 New Best — Trial #{trial_number}  |  Score {score:.4f}",
            "color": color,
            "fields": [
                {
                    "name": "Entry Thresholds",
                    "value": (
                        f"MOM:  Z≥{p.get('mom_vwap_z_min', 0):.1f} / VR≥{p.get('mom_vol_ratio_min', 0):.1f} / "
                        f"DR≥{p.get('mom_delta_min', 0):.2f}\n"
                        f"REV:  Z≤{p.get('rev_vwap_z_max', 0):.1f} / VS≥{p.get('rev_vol_spike_min', 0):.1f}"
                    ),
                    "inline": False,
                },
                {
                    "name": "Risk / Hold",
                    "value": (
                        f"Stop R: {p.get('stop_r', 0):.1f}  Target R: {p.get('target_r', 0):.1f}  "
                        f"R:R = {rr_ratio}×\n"
                        f"Risk/trade: {p.get('risk_per_trade', 0)*100:.1f}%"
                    ),
                    "inline": False,
                },
                {
                    "name": f"Period Breakdown  (Total trades: {total_trades})",
                    "value": periods_text,
                    "inline": False,
                },
            ],
            "footer": {"text": f"Optimizer • {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"},
        }

        payload = {
            "username": "War-Room Optimizer",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
            "embeds": [embed],
        }
        response = requests.post(
            DISCORD_WEBHOOK_URL, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code not in (200, 204):
            logger.error(f"Optimizer Discord notify failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"send_optimizer_best failed: {e}")


def _build_embeds(state: dict, cycle: int) -> list:
    """Builds the rich embeds for the Discord message."""
    embeds = []
    
    timestamp = state.get("cycle_timestamp", "")[:19]
    stance = state.get("market_stance", "Unknown").upper()
    vix = state.get("vix_level", -1)
    
    # Stance color mapping for embed strip
    color = 0x808080 # default gray
    if "BULL" in stance:
        color = 0x00FF00 # Green
    elif "BEAR" in stance or "DEFENSIVE" in stance:
        color = 0xFF0000 # Red
    elif "NEUTRAL" in stance:
        color = 0xFFFF00 # Yellow

    # Overview Embed
    overview_embed = {
        "title": f"📊 Cycle #{cycle} Complete",
        "color": color,
        "description": f"**Stance:** {stance} | **VIX:** {vix:.1f} | **Time:** {timestamp}",
        "fields": []
    }
    
    # Portfolio
    portfolio = state.get("portfolio", {})
    equity = portfolio.get("equity", 0)
    cash = portfolio.get("cash", 0)
    overview_embed["fields"].append({
        "name": "💰 Portfolio",
        "value": f"Equity: ${equity:,.2f}\nCash: ${cash:,.2f}",
        "inline": True
    })
    
    # Risk
    risk_summary = state.get("risk_summary", "No risk data")
    overview_embed["fields"].append({
        "name": "🛡️ Risk Status",
        "value": risk_summary,
        "inline": True
    })

    embeds.append(overview_embed)

    # Fundamentals / Insights Embed
    fund_data = state.get("fundamentals_data", {})
    fund_summary = fund_data.get("summary", "")
    if fund_summary:
        embeds.append({
            "title": "🧠 Research Insights",
            "color": 0x3498db, # Blue
            "description": fund_summary[:2048] # Discord limit
        })

    # Actionable Signals / Executions Embed
    executed = state.get("executed_trades", [])
    if executed:
        exec_desc = ""
        for t in executed:
            icon = "✅" if "filled" in str(t.get('status', '')).lower() else "⚠️"
            exec_desc += f"{icon} **{t.get('side', '').upper()}** {t.get('qty', '')} **{t.get('ticker', '')}** — {t.get('status', '')}\n"
            
        embeds.append({
            "title": "⚡ Executions",
            "color": 0xe67e22, # Orange
            "description": exec_desc[:2048]
        })
        
    # Current Positions Embed
    positions = state.get("positions", [])
    if positions:
        pos_desc = ""
        for p in positions:
            pl = p.get('unrealized_pl', 0)
            pl_pct = p.get('unrealized_pl_pct', 0)
            icon = "🟢" if pl >= 0 else "🔴"
            pos_desc += f"{icon} **{p.get('ticker', '')}**: {p.get('qty', 0)} shares @ ${p.get('current_price', 0):.2f} (P&L: ${pl:.2f} / {pl_pct:+.2f}%)\n"
            
        embeds.append({
            "title": "📈 Current Positions",
            "color": 0x9b59b6, # Purple
            "description": pos_desc[:2048]
        })

    return embeds
