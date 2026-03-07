import os
import requests
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

def test_news():
    print("--- Testing Alpaca SDK News ---")
    try:
        client = NewsClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
        # some versions want a string, some want a list. Try string first based on error.
        req = NewsRequest(symbols="AAPL", limit=3)
        news = client.get_news(req)
        for article in news.news:
            print(f"Title: {article.headline}")
            print(f"URL: {article.url}")
    except Exception as e:
        print(f"SDK Error: {e}")
        print("Falling back to REST API for news...")
        url = "https://data.alpaca.markets/v1beta1/news?symbols=AAPL&limit=3"
        headers = {"Apca-Api-Key-Id": ALPACA_API_KEY, "Apca-Api-Secret-Key": ALPACA_SECRET_KEY}
        resp = requests.get(url, headers=headers)
        print(resp.json())

def test_movers():
    print("\n--- Testing Alpaca Movers REST API ---")
    url = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
    headers = {"Apca-Api-Key-Id": ALPACA_API_KEY, "Apca-Api-Secret-Key": ALPACA_SECRET_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        gainers = data.get("gainers", [])
        print("Top Gainers:")
        for g in gainers[:3]:
            print(f"{g.get('symbol')} - Change: {g.get('percent_change')}%")
    else:
        print(f"Movers API error: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    test_news()
    test_movers()
