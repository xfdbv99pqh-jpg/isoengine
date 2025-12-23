"""Kalshi API crawler for market data."""

import logging
import hashlib
import hmac
import time
from datetime import datetime
from typing import Optional

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class KalshiCrawler(BaseCrawler):
    """Crawler for Kalshi prediction market data."""

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
        self.api_secret = config.kalshi_api_secret
        self.base_url = config.kalshi_demo_url if config.use_kalshi_demo else config.kalshi_base_url
        self.token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

    def is_available(self) -> bool:
        """Check if Kalshi API is configured."""
        return bool(self.api_key and self.api_secret)

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers for Kalshi API."""
        if not self.api_key:
            return {}

        timestamp = str(int(time.time() * 1000))

        # For simple API key auth
        return {
            "Authorization": f"Bearer {self.token}" if self.token else "",
        }

    def _login(self) -> bool:
        """Authenticate with Kalshi API."""
        if not self.is_available():
            logger.warning("Kalshi API credentials not configured")
            return False

        try:
            response = self.post(
                f"{self.base_url}/login",
                json={
                    "email": self.api_key,
                    "password": self.api_secret,
                },
            )
            data = response.json()
            self.token = data.get("token")
            return bool(self.token)
        except Exception as e:
            logger.error(f"Kalshi login failed: {e}")
            return False

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

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get orderbook for a market."""
        params = {"depth": depth}
        headers = self._get_auth_headers()
        response = self.get(f"{self.base_url}/markets/{ticker}/orderbook", params=params, headers=headers)
        return response.json()

    def crawl(self) -> list[CrawlResult]:
        """Crawl all open markets and return standardized results."""
        results = []
        now = datetime.utcnow()

        try:
            # Get all open markets
            markets_response = self.get_markets(status="open", limit=200)
            markets = markets_response.get("markets", [])

            for market in markets:
                # Extract key pricing info
                yes_price = market.get("yes_price", 0) / 100 if market.get("yes_price") else None
                no_price = market.get("no_price", 0) / 100 if market.get("no_price") else None

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
                        "no_price": no_price,
                        "yes_bid": market.get("yes_bid", 0) / 100 if market.get("yes_bid") else None,
                        "yes_ask": market.get("yes_ask", 0) / 100 if market.get("yes_ask") else None,
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

        # Economic indicators
        econ_keywords = ["inflation", "cpi", "gdp", "unemployment", "fed", "rate", "jobs", "payroll"]
        if any(kw in title for kw in econ_keywords) or ticker.startswith(("CPI", "GDP", "FED", "RATE")):
            return "economics"

        # Politics
        political_keywords = ["election", "president", "congress", "senate", "vote", "trump", "biden", "democrat", "republican"]
        if any(kw in title for kw in political_keywords):
            return "politics"

        # Tech
        tech_keywords = ["tesla", "apple", "google", "microsoft", "meta", "nvidia", "ai", "tech"]
        if any(kw in title for kw in tech_keywords):
            return "tech"

        # Weather
        weather_keywords = ["temperature", "hurricane", "weather", "climate"]
        if any(kw in title for kw in weather_keywords):
            return "weather"

        return "other"

    def get_markets_by_category(self, category: str) -> list[dict]:
        """Get markets filtered by category."""
        all_results = self.crawl()
        return [r.data for r in all_results if r.category == category]
