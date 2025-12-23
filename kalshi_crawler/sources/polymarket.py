"""Polymarket API crawler for cross-market comparison."""

import logging
from datetime import datetime
from typing import Optional

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class PolymarketCrawler(BaseCrawler):
    """Crawler for Polymarket prediction market data."""

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="polymarket",
            rate_limit=config.polymarket_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config
        self.base_url = config.polymarket_base_url

    def is_available(self) -> bool:
        """Polymarket public API doesn't require authentication."""
        return True

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict]:
        """Fetch markets from Polymarket."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        response = self.get(f"{self.base_url}/markets", params=params)
        return response.json()

    def get_market(self, condition_id: str) -> dict:
        """Get details for a specific market."""
        response = self.get(f"{self.base_url}/markets/{condition_id}")
        return response.json()

    def search_markets(self, query: str, limit: int = 50) -> list[dict]:
        """Search for markets matching a query."""
        all_markets = self.get_markets(limit=500)  # Get a large batch to search

        query_lower = query.lower()
        matching = []

        for market in all_markets:
            title = market.get("question", "").lower()
            description = market.get("description", "").lower()

            if query_lower in title or query_lower in description:
                matching.append(market)

            if len(matching) >= limit:
                break

        return matching

    def crawl(self) -> list[CrawlResult]:
        """Crawl active Polymarket markets."""
        results = []
        now = datetime.utcnow()

        try:
            markets = self.get_markets(limit=200, active=True)

            for market in markets:
                # Extract outcome prices
                outcomes = market.get("outcomes", [])
                outcome_prices = market.get("outcomePrices", [])

                prices = {}
                if outcomes and outcome_prices:
                    for i, outcome in enumerate(outcomes):
                        if i < len(outcome_prices):
                            try:
                                prices[outcome] = float(outcome_prices[i])
                            except (ValueError, TypeError):
                                pass

                result = CrawlResult(
                    source="polymarket",
                    category=self._categorize_market(market),
                    data_type="market",
                    timestamp=now,
                    data={
                        "condition_id": market.get("conditionId"),
                        "question": market.get("question"),
                        "description": market.get("description", "")[:500],  # Truncate long descriptions
                        "outcomes": outcomes,
                        "prices": prices,
                        "volume": market.get("volume"),
                        "liquidity": market.get("liquidity"),
                        "end_date": market.get("endDate"),
                        "active": market.get("active"),
                    },
                    raw_response=market,
                    metadata={
                        "slug": market.get("slug"),
                        "image": market.get("image"),
                    },
                )
                results.append(result)

            self.last_crawl = now
            logger.info(f"Polymarket crawl complete: {len(results)} markets")

        except Exception as e:
            logger.error(f"Polymarket crawl failed: {e}")
            self.error_count += 1

        return results

    def _categorize_market(self, market: dict) -> str:
        """Categorize market based on content."""
        question = market.get("question", "").lower()
        description = market.get("description", "").lower()
        text = f"{question} {description}"

        # Politics
        political_keywords = ["election", "president", "congress", "senate", "trump", "biden", "vote", "democratic", "republican", "governor"]
        if any(kw in text for kw in political_keywords):
            return "politics"

        # Economics
        econ_keywords = ["inflation", "gdp", "fed", "interest rate", "unemployment", "recession", "cpi"]
        if any(kw in text for kw in econ_keywords):
            return "economics"

        # Tech
        tech_keywords = ["tesla", "spacex", "apple", "google", "microsoft", "ai", "crypto", "bitcoin", "ethereum"]
        if any(kw in text for kw in tech_keywords):
            return "tech"

        # Sports
        sports_keywords = ["nfl", "nba", "mlb", "super bowl", "world series", "championship"]
        if any(kw in text for kw in sports_keywords):
            return "sports"

        return "other"

    def find_similar_markets(self, kalshi_market: dict) -> list[dict]:
        """Find Polymarket markets similar to a Kalshi market."""
        title = kalshi_market.get("title", "")

        # Extract key terms
        keywords = self._extract_keywords(title)

        matches = []
        for keyword in keywords:
            results = self.search_markets(keyword, limit=10)
            matches.extend(results)

        # Deduplicate
        seen = set()
        unique_matches = []
        for market in matches:
            cid = market.get("conditionId")
            if cid and cid not in seen:
                seen.add(cid)
                unique_matches.append(market)

        return unique_matches[:20]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction
        stopwords = {"the", "a", "an", "is", "are", "will", "be", "to", "of", "in", "on", "at", "by", "for"}

        words = text.lower().replace("?", "").replace("!", "").split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return keywords[:5]  # Top 5 keywords
