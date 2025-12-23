"""Kalshi API crawler for market data."""

import logging
import time
from datetime import datetime
from typing import Optional

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class KalshiCrawler(BaseCrawler):
    """Crawler for Kalshi prediction market data."""

    # Kalshi public API - no auth needed for market data
    PUBLIC_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="kalshi",
            rate_limit=config.kalshi_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config
        self.api_key = config.kalshi_api_key
        # Use public API for market data (no auth required)
        self.base_url = self.PUBLIC_API_URL

    def is_available(self) -> bool:
        """Kalshi public market data is always available."""
        return True  # Public endpoints don't require auth

    def _get_auth_headers(self) -> dict:
        """Generate headers for Kalshi API."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
    ) -> dict:
        """Fetch markets from Kalshi API."""
        params = {
            "status": status,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker

        headers = self._get_auth_headers()
        response = self.get(f"{self.base_url}/markets", params=params, headers=headers)
        return response.json()

    def get_market(self, ticker: str) -> dict:
        """Get details for a specific market."""
        headers = self._get_auth_headers()
        response = self.get(f"{self.base_url}/markets/{ticker}", headers=headers)
        return response.json()

    def get_events(self, status: str = "open", limit: int = 100) -> dict:
        """Fetch events from Kalshi API."""
        params = {"status": status, "limit": limit}
        headers = self._get_auth_headers()
        response = self.get(f"{self.base_url}/events", params=params, headers=headers)
        return response.json()

    def get_series(self, limit: int = 100) -> dict:
        """Fetch series from Kalshi API."""
        params = {"limit": limit}
        headers = self._get_auth_headers()
        response = self.get(f"{self.base_url}/series", params=params, headers=headers)
        return response.json()

    def crawl(self) -> list[CrawlResult]:
        """Crawl all open markets and return standardized results."""
        results = []
        now = datetime.utcnow()

        try:
            # Get all open markets with pagination
            all_markets = []
            cursor = None

            for _ in range(10):  # Max 10 pages (1000 markets)
                markets_response = self.get_markets(status="open", limit=100, cursor=cursor)
                markets = markets_response.get("markets", [])
                all_markets.extend(markets)

                cursor = markets_response.get("cursor")
                if not cursor or not markets:
                    break

            logger.info(f"Fetched {len(all_markets)} Kalshi markets")

            for market in all_markets:
                # Extract key pricing info - Kalshi returns prices in cents
                yes_bid = market.get("yes_bid")
                yes_ask = market.get("yes_ask")

                # Calculate mid price
                if yes_bid is not None and yes_ask is not None:
                    yes_price = (yes_bid + yes_ask) / 2 / 100
                elif yes_bid is not None:
                    yes_price = yes_bid / 100
                elif yes_ask is not None:
                    yes_price = yes_ask / 100
                else:
                    yes_price = None

                result = CrawlResult(
                    source="kalshi",
                    category=self._categorize_market(market),
                    data_type="market",
                    timestamp=now,
                    data={
                        "ticker": market.get("ticker"),
                        "title": market.get("title"),
                        "subtitle": market.get("subtitle"),
                        "yes_price": yes_price,
                        "yes_bid": yes_bid / 100 if yes_bid else None,
                        "yes_ask": yes_ask / 100 if yes_ask else None,
                        "volume": market.get("volume"),
                        "volume_24h": market.get("volume_24h"),
                        "open_interest": market.get("open_interest"),
                        "close_time": market.get("close_time"),
                        "expiration_time": market.get("expiration_time"),
                        "status": market.get("status"),
                    },
                    raw_response=market,
                    metadata={
                        "event_ticker": market.get("event_ticker"),
                        "series_ticker": market.get("series_ticker"),
                    },
                )
                results.append(result)

            self.last_crawl = now
            logger.info(f"Kalshi crawl complete: {len(results)} markets")

        except Exception as e:
            logger.error(f"Kalshi crawl failed: {e}")
            self.error_count += 1

        return results

    def _categorize_market(self, market: dict) -> str:
        """Categorize market based on ticker/title."""
        ticker = market.get("ticker", "").upper()
        title = market.get("title", "").lower()
        event_ticker = market.get("event_ticker", "").upper()

        # Economic indicators
        econ_keywords = ["inflation", "cpi", "gdp", "unemployment", "fed", "rate", "jobs", "payroll", "fomc"]
        if any(kw in title for kw in econ_keywords) or any(x in ticker or x in event_ticker for x in ["CPI", "GDP", "FED", "RATE", "FOMC", "ECON"]):
            return "economics"

        # Politics
        political_keywords = ["election", "president", "congress", "senate", "vote", "trump", "biden", "democrat", "republican", "governor", "electoral"]
        if any(kw in title for kw in political_keywords):
            return "politics"

        # Tech
        tech_keywords = ["tesla", "apple", "google", "microsoft", "meta", "nvidia", "ai", "spacex", "twitter"]
        if any(kw in title for kw in tech_keywords):
            return "tech"

        # Weather
        weather_keywords = ["temperature", "hurricane", "weather", "climate", "storm"]
        if any(kw in title for kw in weather_keywords):
            return "weather"

        # Crypto
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto"]
        if any(kw in title for kw in crypto_keywords):
            return "crypto"

        return "other"

    def get_markets_by_category(self, category: str) -> list[dict]:
        """Get markets filtered by category."""
        all_results = self.crawl()
        return [r.data for r in all_results if r.category == category]

    def search_markets(self, query: str) -> list[dict]:
        """Search markets by keyword."""
        all_results = self.crawl()
        query_lower = query.lower()
        return [
            r.data for r in all_results
            if query_lower in (r.data.get("title", "") or "").lower()
            or query_lower in (r.data.get("ticker", "") or "").lower()
        ]
