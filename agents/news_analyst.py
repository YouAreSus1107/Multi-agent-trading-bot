"""
News Analyst Agent (Slow Loop — every 5 min)
Monitors news, discovers niche tickers, maintains a continuous reasoning log.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_factory import get_llm
from utils.logger import get_logger
from services.news_service import NewsService
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
import json
import os
import requests
from datetime import datetime, timezone

logger = get_logger("news_analyst")

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "news_analyst_log.jsonl")
SEEN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "seen_headlines.json")

SYSTEM_PROMPT = """You are the News Analyst of a day-trading bot. You are an expert at finding trading opportunities hidden in news.

Your job:
1. Read the news and identify which SPECIFIC STOCKS will be affected TODAY.
2. Discover NICHE/SPECIFIC tickers — not just big ETFs. Examples:
   - "Earnings beat, raised guidance" → find that specific ticker
   - "FDA grants approval" → find that specific biotech (RLAY, MRNA, ARWR, etc.)
   - "Pentagon awards $500M cyber contract" → PLTR, PANW, CRWD
   - "AI chip demand surges" → NVDA, AMD, SMCI, AVGO
   - "Bitcoin hits new high" → MSTR, MARA, COIN, RIOT
   - "Iran threatens Strait of Hormuz" → OXY, XOM, LMT, RTX
   - "Short squeeze candidate high volume" → find the actual ticker
3. Assess the MAGNITUDE of each event's impact (1-10 scale).
4. Think about SECOND-ORDER effects (e.g., oil disruption → airline costs up → UAL down).
5. Write a brief reasoning log.

SECTOR DIVERSITY RULE (CRITICAL):
- You MUST cover AT LEAST 2 DIFFERENT sectors per analysis.
- If recent cycles were dominated by energy/defense (Iran war), you MUST find tickers in OTHER sectors:
  tech, biotech, crypto, consumer, industrial, small-cap momentum, etc.
- DO NOT output only OXY/XOM/LMT/RTX if there are other newsworthy events in the feed.
- Missing a biotech catalyst or earnings beat because you fixated on geopolitics is a BAD analysis.

ALGORITHMIC MOMENTUM RULE:
- You will be provided with the current Top Gainers and Losers from the market screener. 
- You MUST evaluate these symbols alongside the news. If a stock is up 150% on massive volume, include it in your `niche_tickers_discovered` list even if the news is sparse. Combine the quant momentum with whatever catalyst you can find.

Focus on stocks that will MOVE TODAY — not long-term plays.

YOU MUST RESPOND WITH VALID JSON ONLY.

{
    "events": [
        {
            "headline": "short description",
            "impact": 8,
            "affected_tickers": ["PLTR", "CRWD"],
            "direction": "bullish",
            "reasoning": "Why these tickers will move",
            "deep_context": [
                "Bullet 1: In-depth detail about the news and source...",
                "Bullet 2: Specific numbers, earnings stats, or contract values...",
                "Bullet 3: Historical correlation or similar past events...",
                "Bullet 4: Potential macro or geopolitical ripple effects...",
                "Bullet 5: Reddit/Social sentiment surrounding this event if applicable...",
                "... (provide up to 8-10 rich, detailed bullet points per event so the Research Team has maximum context)"
            ]
        }
    ],
    "niche_tickers_discovered": ["PLTR", "RKLB", "IONQ"],
    "sector_impact": {
        "defense": {"sentiment": 0.85, "direction": "bullish"},
        "tech": {"sentiment": 0.40, "direction": "bearish"}
    },
    "market_mood": "risk-off",
    "thinking_log": "My reasoning process summary covering MULTIPLE sectors..."
}
"""


class NewsAnalyst:
    """
    Slow-loop analyst: deep news analysis with niche ticker discovery.
    Maintains continuous log of reasoning at data/news_analyst_log.jsonl.
    """

    def __init__(self):
        self.llm = get_llm("parsing")
        self.news_service = NewsService()
        self.seen_urls = self._load_seen_urls()
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def analyze(self) -> dict:
        """
        Fetch news and perform deep analysis.
        Writes reasoning to continuous log file.
        """
        # Fetch fresh news
        all_news_items = self.news_service.fetch_all_news(max_results=30)
        
        # Deduplicate
        new_items = []
        for item in all_news_items:
            url = item.get("url", "")
            if not url or url in self.seen_urls:
                continue
            self.seen_urls.add(url)
            new_items.append(item)

        if not new_items:
            logger.info("News Analyst: No new headlines since last cycle")
            return self._empty_result("No new headlines/events discovered")

        # Persist updated seen URLs immediately so we don't re-process
        self._save_seen_urls()

        # Format news for LLM
        news_text = "\n".join([
            f"- [{item.get('source', 'unknown')}] {item.get('title', '')} "
            f"(relevance: {item.get('relevance_score', 0):.0%})"
            for item in new_items[:20]
        ])

        # Build anti-fixation context from recent log
        recent_sectors = []
        recent_log = self.get_recent_log(n=3)
        for entry in recent_log:
            for sector in entry.get("sector_breakdown", []):
                recent_sectors.append(sector)
        anti_fixation = ""
        if recent_sectors:
            sector_counts: dict = {}
            for s in recent_sectors:
                sector_counts[s] = sector_counts.get(s, 0) + 1
            dominant = [s for s, c in sector_counts.items() if c >= 2]
            if dominant:
                anti_fixation = (
                    f"\n\nANTI-FIXATION WARNING: Recent cycles were dominated by {', '.join(dominant)}. "
                    "You MUST discover tickers from DIFFERENT sectors this cycle. "
                    "Do not output only the same tickers as recent cycles."
                )

        movers_text = self._fetch_top_movers()

        prompt = f"""Analyze these news items AND the top market movers for DAY TRADING opportunities.
Find specific, niche tickers that will move TODAY.
You MUST cover at least 2 different sectors. Do not fixate on one theme.
{anti_fixation}

NEWS FEED:
{news_text}

MARKET MOVERS (Screener):
{movers_text}

Remember: We day-trade volatile individual stocks (PLTR, MSTR, SMCI, IONQ, NVDA, AMD, LMT, etc.)
NOT ETFs like XLE or GLD. Find the specific companies affected."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = self._parse(response.content)

            # Write to continuous log
            self._write_log(result, new_items)

            logger.info(
                f"News Analyst: {len(result.get('events', []))} events, "
                f"{len(result.get('niche_tickers_discovered', []))} niche tickers"
            )
            result["news_items"] = new_items
            return result

        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return self._empty_result(str(e))

    def _write_log(self, result: dict, news_items: list):
        """Append reasoning to the continuous log file."""
        try:
            sectors = list(result.get("sector_impact", {}).keys())
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "events_count": len(result.get("events", [])),
                "events": result.get("events", []),
                "niche_tickers": result.get("niche_tickers_discovered", []),
                "sector_breakdown": sectors,
                "market_mood": result.get("market_mood", "unknown"),
                "thinking_log": result.get("thinking_log", ""),
                "sector_impact": result.get("sector_impact", {}),
                "news_headlines": [n.get("title", "")[:120] for n in news_items[:5]],
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
            # Also persist seen URLs after successful analysis
            self._save_seen_urls()
        except Exception as e:
            logger.warning(f"Failed to write news log: {e}")

    def get_recent_log(self, n: int = 5) -> list[dict]:
        """Read last N log entries for context."""
        try:
            if not os.path.exists(LOG_FILE):
                return []
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
            return entries
        except Exception:
            return []

    def _load_seen_urls(self) -> set:
        """Load persisted seen urls from disk."""
        try:
            if os.path.exists(SEEN_FILE):
                with open(SEEN_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("urls", []))
        except Exception as e:
            logger.warning(f"Failed to load seen urls: {e}")
        return set()

    def _save_seen_urls(self):
        """Persist seen urls to disk — pretty-printed with metadata header."""
        try:
            urls_list = sorted(self.seen_urls)[-500:]
            data = {
                "_meta": {
                    "count": len(urls_list),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "note": "Seen URL de-duplication cache. Last 500 entries kept."
                },
                "urls": urls_list,
            }
            with open(SEEN_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save seen urls: {e}")

    def _empty_result(self, reason: str) -> dict:
        return {
            "events": [],
            "niche_tickers_discovered": [],
            "sector_impact": {},
            "market_mood": "unknown",
            "thinking_log": reason,
            "news_items": [],
        }

    def _parse(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return self._empty_result("JSON parse error")

    def _fetch_top_movers(self) -> str:
        """Fetch top gainers and losers from Alpaca Screener API."""
        try:
            url = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
            headers = {
                "Apca-Api-Key-Id": ALPACA_API_KEY,
                "Apca-Api-Secret-Key": ALPACA_SECRET_KEY,
                "Accept": "application/json"
            }
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code != 200:
                logger.warning(f"Screener API failed: {resp.status_code}")
                return "Screener data unavailable."
                
            data = resp.json()
            gainers = data.get("gainers", [])[:5]
            losers = data.get("losers", [])[:5]
            
            lines = ["[Gainers]"]
            for g in gainers:
                lines.append(f"- {g.get('symbol')}: +{g.get('percent_change', 0):.1f}% (Vol: {g.get('volume', 0)})")
                
            lines.append("\n[Losers]")
            for l in losers:
                lines.append(f"- {l.get('symbol')}: {l.get('percent_change', 0):.1f}% (Vol: {l.get('volume', 0)})")
                
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Error fetching top movers: {e}")
            return "Screener data unavailable."
