import os
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
