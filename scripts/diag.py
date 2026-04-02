"""
Quick diagnostic test for the new day-trading agents.
Tests: Technical Analyst + News Analyst + LLM failover.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import traceback

print("=" * 60)
print("DAY TRADING BOT -- DIAGNOSTIC TEST")
print("=" * 60)

# 1. Fetch real news
print("\n[STEP 1] Fetching news via NewsService...")
from services.news_service import NewsService
ns = NewsService()
news = ns.fetch_all_news(max_results=5)
print(f"  Got {len(news)} items")
for i, item in enumerate(news[:3]):
    print(f"  [{i+1}] {item.get('title', 'No title')[:80]}")

# 2. Run Technical Analyst (no LLM needed)
print("\n[STEP 2] Running Technical Analyst (pure math)...")
from agents.technical_analyst import TechnicalAnalyst
tech = TechnicalAnalyst()
try:
    result = tech.analyze(["PLTR", "NVDA", "MSTR"])
    for ticker, data in result.items():
        pro = data.get("pro_signals", {})
        print(f"  {ticker}: trend={data.get('score', 0)}, rsi={data.get('rsi', 0):.1f}, "
              f"signals={data.get('signals', [])}")
        print(f"    PRO: ema100={pro.get('trend_bias', {}).get('trend_bias', '?')}, "
              f"vwap_dist={pro.get('vwap', {}).get('distance_pct', 0):.1f}%, "
              f"stop_hunt_L={pro.get('stop_hunt', {}).get('stop_hunt_long', False)}, "
              f"parabolic={pro.get('parabolic', {}).get('direction', 'none')}")
except Exception as e:
    print(f"  [ERROR] Technical Analyst failed:")
    traceback.print_exc()

# 3. Test LLM failover
print("\n[STEP 3] Testing LLM failover...")
from utils.llm_factory import get_llm
from langchain_core.messages import HumanMessage
try:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Say 'LLM is working' and nothing else.")])
    print(f"  LLM says: {response.content}")
except Exception as e:
    print(f"  [ERROR] LLM failed:")
    traceback.print_exc()

# 4. Test News Analyst (uses LLM)
print("\n[STEP 4] Running News Analyst...")
from agents.news_analyst import NewsAnalyst
try:
    na = NewsAnalyst()
    result = na.analyze()
    print(f"  Events: {len(result.get('events', []))}")
    print(f"  Niche tickers: {result.get('niche_tickers_discovered', [])}")
    print(f"  Market mood: {result.get('market_mood', '?')}")
except Exception as e:
    print(f"  [ERROR] News Analyst failed:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 60)
