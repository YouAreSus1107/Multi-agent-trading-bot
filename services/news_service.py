"""
War-Room Bot -- News Ingestion Service (Multi-Source)
Sources: Tavily (war + market), Polymarket (predictions), Google News RSS (free world news)
"""

from tavily import TavilyClient
from config import TAVILY_API_KEY, WAR_KEYWORDS, MARKET_KEYWORDS, CATALYST_KEYWORDS, ALPACA_API_KEY, ALPACA_SECRET_KEY
from utils.logger import get_logger
from utils.rate_limiter import retry_on_rate_limit
import urllib.request
import json
import requests
import xml.etree.ElementTree as ET

logger = get_logger("news_service")


class NewsService:
    """Multi-source news aggregation for trading intelligence."""

    def __init__(self):
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.polymarket_url = "https://gamma-api.polymarket.com/events?closed=false&limit=15"

    def fetch_all_news(self, max_results: int = 15) -> list[dict]:
        """
        Fetch news from ALL sources, covering war AND general market.
        Returns sorted by relevance.
        """
        all_news = []
        self._tavily_exhausted = getattr(self, "_tavily_exhausted", False)

        # Source 1: Tavily geopolitical/war news
        if not self._tavily_exhausted:
            all_news.extend(self._fetch_tavily_war(max_results=6))

        # Source 2: Tavily general market news
        if not self._tavily_exhausted:
            all_news.extend(self._fetch_tavily_market(max_results=8))

        # Source 3: Catalyst/earnings/momentum news
        if not self._tavily_exhausted:
            all_news.extend(self._fetch_tavily_catalysts(max_results=10))

        # Fallback: Alpaca News (if Tavily is exhausted)
        if self._tavily_exhausted:
            logger.info("Tavily is exhausted. Falling back to Alpaca News API.")
            all_news.extend(self._fetch_alpaca_news_fallback(max_results=20))

        # Source 4: Polymarket predictions
        poly_max = 20 if self._tavily_exhausted else 10
        all_news.extend(self._fetch_polymarket(max_results=poly_max))

        # Source 5: Google News RSS
        google_slots = 3 if self._tavily_exhausted else 1
        all_news.extend(self._fetch_google_news_rss(num_slots=google_slots))

        # Source 6: Reddit chatter (WallStreetBets, Stocks)
        all_news.extend(self._fetch_reddit(max_results=10))

        # Deduplicate
        seen_titles = set()
        unique_news = []
        for item in all_news:
            key = item.get("title", "")[:40].lower().strip()
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique_news.append(item)

        unique_news.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        logger.info(
            f"Fetched {len(unique_news)} unique news items",
            extra={"agent": "news_service", "action": "fetch",
                   "data": {"count": len(unique_news)}},
        )
        return unique_news[:max_results]

    # Keep backwards-compatible method name
    def fetch_war_news(self, max_results: int = 10) -> list[dict]:
        return self.fetch_all_news(max_results)

    def _fetch_tavily_war(self, max_results: int) -> list[dict]:
        """Fetch war/geopolitical news via Tavily."""
        try:
            query = " OR ".join(WAR_KEYWORDS[:8])
            response = self.tavily_client.search(
                query=f"latest news: {query}",
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )
            return [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "source": self._extract_source(r.get("url", "")),
                    "relevance_score": r.get("score", 0),
                    "category": "geopolitical",
                    "provider": "tavily",
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            msg = str(e).lower()
            if "usage limit" in msg or "plan" in msg or "exceed" in msg:
                self._tavily_exhausted = True
                logger.warning(f"Tavily API limit reached (disabling for session): {e}")
            else:
                logger.warning(f"Tavily war fetch failed: {e}")
            return []

    def _fetch_tavily_market(self, max_results: int) -> list[dict]:
        """Fetch general market news via Tavily."""
        try:
            query = " OR ".join(MARKET_KEYWORDS[:6])
            response = self.tavily_client.search(
                query=f"today market news: {query}",
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )
            return [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "source": self._extract_source(r.get("url", "")),
                    "relevance_score": r.get("score", 0) * 0.8,  # Slightly lower weight
                    "category": "market",
                    "provider": "tavily",
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            msg = str(e).lower()
            if "usage limit" in msg or "plan" in msg or "exceed" in msg:
                self._tavily_exhausted = True
                logger.warning(f"Tavily API limit reached (disabling for session): {e}")
            else:
                logger.warning(f"Tavily market fetch failed: {e}")
            return []

    def _fetch_tavily_catalysts(self, max_results: int = 6) -> list[dict]:
        """Fetch catalyst/momentum events: earnings beats, FDA, contracts, short squeezes."""
        try:
            query = " OR ".join(CATALYST_KEYWORDS[:6])
            response = self.tavily_client.search(
                query=f"today: {query} stock",
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )
            return [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "source": self._extract_source(r.get("url", "")),
                    "relevance_score": r.get("score", 0) * 0.9,
                    "category": "catalyst",
                    "provider": "tavily",
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            msg = str(e).lower()
            if "usage limit" in msg or "plan" in msg or "exceed" in msg:
                self._tavily_exhausted = True
                logger.warning(f"Tavily API limit reached (disabling for session): {e}")
            else:
                logger.warning(f"Tavily catalyst fetch failed: {e}")
            return []

    @retry_on_rate_limit(max_retries=3, initial_backoff=2.0)
    def _fetch_alpaca_news_fallback(self, max_results: int = 20) -> list[dict]:
        """Fallback to Alpaca's Market News API when Tavily is rate-limited."""
        try:
            url = f"https://data.alpaca.markets/v1beta1/news?limit={max_results}"
            headers = {
                "Apca-Api-Key-Id": ALPACA_API_KEY,
                "Apca-Api-Secret-Key": ALPACA_SECRET_KEY,
                "Accept": "application/json"
            }
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 429:
                raise Exception("Alpaca API rate limit HTTP 429")
            elif resp.status_code != 200:
                logger.warning(f"Alpaca news fallback failed: {resp.status_code} - {resp.text}")
                return []
                
            data = resp.json()
            news_items = []
            
            for article in data.get("news", []):
                symbols = article.get("symbols", [])
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                if symbols:
                    summary = f"[{','.join(symbols)}] {summary}"
                    
                news_items.append({
                    "title": headline,
                    "content": summary,
                    "url": article.get("url", ""),
                    "source": article.get("source", "alpaca"),
                    "relevance_score": 0.8,
                    "category": "market",
                    "provider": "alpaca",
                })
                
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Alpaca news fallback: {e}")
            return []

    def _fetch_polymarket(self, max_results: int = 15) -> list[dict]:
        """Fetch prediction market signals from Polymarket."""
        try:
            # We construct a URL with limit applied to fetch
            url = self.polymarket_url.replace("limit=15", f"limit={max_results * 2}") # grab more to filter
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "WarRoomBot/2.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            news_items = []
            for event in data[:max_results]:
                title = event.get("title", "")
                description = event.get("description", "")
                combined = (title + " " + description).lower()

                # Accept broader events (not just war)
                relevant_keywords = [
                    "iran", "war", "strike", "oil", "military", "nuclear",
                    "tariff", "trade", "recession", "fed", "inflation",
                    "china", "russia", "election", "stock", "market",
                    "sanctions", "conflict", "crisis",
                ]
                is_relevant = any(kw in combined for kw in relevant_keywords)

                if is_relevant:
                    markets = event.get("markets", [])
                    odds_text = ""
                    if markets:
                        for m in markets[:2]:
                            outcome = m.get("groupItemTitle", m.get("question", ""))
                            prob = m.get("lastTradePrice", 0)
                            if prob:
                                odds_text += f" [Polymarket: {outcome} = {float(prob)*100:.0f}%]"

                    news_items.append({
                        "title": f"[PREDICTION] {title}",
                        "content": description + odds_text,
                        "url": f"https://polymarket.com/event/{event.get('slug', '')}",
                        "source": "polymarket.com",
                        "relevance_score": 0.65,
                        "category": "prediction",
                        "provider": "polymarket",
                    })

            logger.info(f"Polymarket: {len(news_items)} relevant events")
            return news_items

        except Exception as e:
            logger.warning(f"Polymarket fetch failed (non-critical): {e}")
            return []

    def _fetch_google_news_rss(self, num_slots: int = 1) -> list[dict]:
        """
        Fetch from Google News RSS — ROTATING SECTOR FEEDS.
        Cycles through different sectors each call to prevent topic lock.
        Iran/geopolitical feed is ONE of many, not the only one.
        """
        from datetime import datetime
        hour = datetime.now().hour
        news_items = []

        # Each slot targets a DIFFERENT sector to guarantee diversity
        SECTOR_FEEDS = [
            # Slot 0 — Earnings and corporate catalysts
            [
                "https://news.google.com/rss/search?q=earnings+beat+surprise+stock&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=earnings+report+revenue+guidance&hl=en-US&gl=US&ceid=US:en",
            ],
            # Slot 1 — FDA/Biotech/Healthcare catalysts
            [
                "https://news.google.com/rss/search?q=FDA+approval+biotech+drug+stock&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=clinical+trial+results+PDUFA+stock&hl=en-US&gl=US&ceid=US:en",
            ],
            # Slot 2 — AI / Tech / Semiconductors
            [
                "https://news.google.com/rss/search?q=AI+chip+NVDA+AMD+SMCI+stock&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=artificial+intelligence+tech+stock+surge&hl=en-US&gl=US&ceid=US:en",
            ],
            # Slot 3 — Small/Mid-cap momentum and short squeeze
            [
                "https://news.google.com/rss/search?q=small+cap+stock+rally+momentum+high+volume&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=short+squeeze+unusual+volume+stock+runner&hl=en-US&gl=US&ceid=US:en",
            ],
            # Slot 4 — Geopolitical/War (kept but only 1 slot of 6)
            [
                "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=iran+war+oil+military&hl=en-US&gl=US&ceid=US:en",
            ],
            # Slot 5 — Macro / Fed / Crypto / Crypto-adjacent stocks
            [
                "https://news.google.com/rss/search?q=Federal+Reserve+inflation+rate+cut+stock&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=bitcoin+crypto+MSTR+COIN+MARA+stock&hl=en-US&gl=US&ceid=US:en",
            ],
        ]

        for i in range(num_slots):
            slot = ((hour // 2) + i) % 6
            feeds = SECTOR_FEEDS[slot]
            sector_name = ["earnings", "biotech/FDA", "AI/tech", "small-cap/momentum", "geopolitical", "macro/crypto"][slot]
            logger.info(f"Google News RSS: fetching slot {slot} ({sector_name})")

            for feed_url in feeds:
                try:
                    req = urllib.request.Request(
                        feed_url,
                        headers={"User-Agent": "WarRoomBot/2.0"},
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        xml_data = resp.read().decode("utf-8")

                    root = ET.fromstring(xml_data)
                    channel = root.find("channel")
                    if channel is None:
                        continue

                    for item in channel.findall("item")[:5]:
                        title = item.findtext("title", "")
                        link = item.findtext("link", "")
                        pub_date = item.findtext("pubDate", "")
                        source_elem = item.find("source")
                        source = source_elem.text if source_elem is not None else "google-news"

                        news_items.append({
                            "title": title,
                            "content": f"{title} (Published: {pub_date})",
                            "url": link,
                            "source": source,
                            "relevance_score": 0.6,
                            "category": sector_name,
                            "provider": "google_news",
                        })

                except Exception as e:
                    logger.warning(f"Google News RSS failed for feed: {e}")

        logger.info(f"Google News RSS [{sector_name}]: {len(news_items)} articles")
        return news_items

    def _fetch_reddit(self, max_results: int = 10) -> list[dict]:
        """Fetch trending posts from WallStreetBets and Stocks subreddits."""
        news_items = []
        subreddits = ["wallstreetbets", "stocks", "investing"]
        try:
            for sub in subreddits:
                # Grab the top posts of the day
                url = f"https://www.reddit.com/r/{sub}/top.json?limit=5&t=day"
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "WarRoomBot/2.0 (by /u/trader)"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    score = post.get("score", 0)
                    upvote_ratio = post.get("upvote_ratio", 0)
                    
                    # Ensure it's a decently popular post
                    if score <= 50 or upvote_ratio <= 0.7:
                        continue
                        
                    title = post.get("title", "")
                    
                    if not post.get("is_self", True) and post.get("url"):
                        link_url = post.get("url", "")
                        text = f"[External Link: {link_url}]\n"
                        try:
                            # Try to fetch and parse the article text
                            resp = requests.get(link_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
                            if resp.status_code == 200:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(resp.text, "html.parser")
                                paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
                                article_text = " ".join([p for p in paragraphs if p]).strip()
                                if len(article_text) > 50:
                                    text += article_text[:400]
                                else:
                                    text += "No readable article text found."
                            else:
                                text += f"Failed to fetch content (HTTP {resp.status_code})."
                        except Exception as e:
                            text += "Failed to load external link due to generic error or timeout."
                    else:
                        text = post.get("selftext", "")
                        # Require at least some meaningful DD/text (e.g., > 100 chars)
                        if len(text) < 100:
                            continue
                        text = text[:400] # Grab first 400 chars of DD
                        
                    content_str = f"{title}\n{text}"
                    
                    # Deduplication URLs: Link posts use the external requested URL, self-posts use the reddit permalink
                    final_url = post.get("url", "") if not post.get("is_self", True) else f"https://reddit.com{post.get('permalink', '')}"
                    
                    news_items.append({
                        "title": f"[Reddit r/{sub}] {title}",
                        "content": content_str,
                        "url": final_url,
                        "source": f"reddit/r/{sub}",
                        "relevance_score": 0.85,
                        "category": "social",
                        "provider": "reddit"
                    })
                        
            logger.info(f"Reddit API: fetched {len(news_items)} trending text posts")
            return news_items[:max_results]
        except Exception as e:
            logger.warning(f"Reddit fetch failed (non-critical): {e}")
            return []

    def _build_query(self) -> str:
        query = " OR ".join(WAR_KEYWORDS[:8])
        return f"latest news: {query}"

    @staticmethod
    def _extract_source(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "unknown"
