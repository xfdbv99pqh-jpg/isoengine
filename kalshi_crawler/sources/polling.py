"""Polling data aggregator for 270toWin, RCP, and election data."""

import logging
import re
import json
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class PollingCrawler(BaseCrawler):
    """Crawler for polling aggregator data."""

    # Working polling data sources
    SOURCES = {
        "270towin": {
            "base": "https://www.270towin.com",
            "endpoints": {
                "polls": "/2024-presidential-election-polls/",
                "map": "/maps/consensus-2024-electoral-map-702",
            }
        },
        "rcp": {
            "base": "https://www.realclearpolling.com",
            "endpoints": {
                "president": "/latest-polls/president/",
                "elections": "/elections/",
            }
        },
        "silver_bulletin": {
            "base": "https://www.natesilver.net",
            "endpoints": {
                "forecast": "/",
            }
        }
    }

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="polling",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config

    def is_available(self) -> bool:
        """Polling sources are publicly available."""
        return True

    def _scrape_270towin(self) -> list[dict]:
        """Scrape 270toWin for electoral data."""
        results = []
        base = self.SOURCES["270towin"]["base"]

        try:
            # Get consensus electoral map
            url = f"{base}/maps/consensus-2024-electoral-map-702"
            response = self.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for electoral vote counts
            result = {
                "source": "270towin",
                "poll_type": "electoral_map",
                "url": url,
                "scraped_at": datetime.utcnow().isoformat(),
            }

            # Try to find vote counts in the page
            # 270towin typically has these in specific elements
            dem_votes = soup.find(class_=re.compile(r"dem|democratic|blue", re.I))
            rep_votes = soup.find(class_=re.compile(r"rep|republican|red", re.I))

            if dem_votes:
                result["dem_text"] = dem_votes.get_text(strip=True)[:100]
            if rep_votes:
                result["rep_text"] = rep_votes.get_text(strip=True)[:100]

            results.append(result)

        except Exception as e:
            logger.error(f"270toWin scrape failed: {e}")

        return results

    def _scrape_rcp_latest(self) -> list[dict]:
        """Scrape RealClearPolling latest polls page."""
        results = []
        base = self.SOURCES["rcp"]["base"]

        try:
            url = f"{base}/elections/"
            response = self.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            result = {
                "source": "rcp",
                "poll_type": "latest_polls",
                "url": url,
                "scraped_at": datetime.utcnow().isoformat(),
            }

            # Look for poll data in tables or lists
            tables = soup.find_all("table")
            poll_data = []

            for table in tables[:3]:  # First 3 tables
                rows = table.find_all("tr")
                for row in rows[:10]:  # First 10 rows
                    cells = row.find_all(["td", "th"])
                    if cells:
                        row_text = " | ".join(c.get_text(strip=True)[:50] for c in cells[:5])
                        if row_text.strip():
                            poll_data.append(row_text)

            result["poll_data"] = poll_data[:20]  # Limit to 20 rows
            result["has_data"] = len(poll_data) > 0
            results.append(result)

        except Exception as e:
            logger.error(f"RCP scrape failed: {e}")

        return results

    def _get_election_json_feeds(self) -> list[dict]:
        """Try to get election data from JSON APIs."""
        results = []

        # Some sites expose JSON endpoints
        json_endpoints = [
            "https://projects.fivethirtyeight.com/polls/president-general/2024/polls.json",
            "https://cdn.split.io/api/splitChanges",
        ]

        for url in json_endpoints:
            try:
                response = self.get(url)
                if response.headers.get("content-type", "").startswith("application/json"):
                    data = response.json()
                    results.append({
                        "source": "json_feed",
                        "url": url,
                        "data": data if isinstance(data, dict) else {"items": data[:50]},
                        "scraped_at": datetime.utcnow().isoformat(),
                    })
            except Exception:
                continue  # Skip failed endpoints silently

        return results

    def crawl(self) -> list[CrawlResult]:
        """Crawl all polling sources."""
        results = []
        now = datetime.utcnow()

        # 270toWin
        try:
            for item in self._scrape_270towin():
                results.append(CrawlResult(
                    source="polling_270towin",
                    category="politics",
                    data_type="poll",
                    timestamp=now,
                    data=item,
                    metadata={"url": item.get("url")},
                ))
        except Exception as e:
            logger.error(f"270toWin crawl failed: {e}")
            self.error_count += 1

        # RCP
        try:
            for item in self._scrape_rcp_latest():
                results.append(CrawlResult(
                    source="polling_rcp",
                    category="politics",
                    data_type="poll",
                    timestamp=now,
                    data=item,
                    metadata={"url": item.get("url")},
                ))
        except Exception as e:
            logger.error(f"RCP crawl failed: {e}")
            self.error_count += 1

        # JSON feeds
        try:
            for item in self._get_election_json_feeds():
                results.append(CrawlResult(
                    source="polling_json",
                    category="politics",
                    data_type="poll",
                    timestamp=now,
                    data=item,
                    metadata={"url": item.get("url")},
                ))
        except Exception as e:
            logger.error(f"JSON feed crawl failed: {e}")
            self.error_count += 1

        self.last_crawl = now
        logger.info(f"Polling crawl complete: {len(results)} results")
        return results

    def get_polling_summary(self) -> dict:
        """Get a summary of current polling data."""
        results = self.crawl()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "sources_crawled": len(results),
            "data": [r.data for r in results],
        }
