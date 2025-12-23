"""Polling data aggregator for 538, RealClearPolitics, etc."""

import logging
import re
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class PollingCrawler(BaseCrawler):
    """Crawler for polling aggregator data."""

    # RealClearPolitics URLs
    RCP_BASE = "https://www.realclearpolling.com"
    RCP_POLLS = {
        "president_approval": "/polls/approval/president/",
        "generic_ballot": "/polls/other/generic_congressional_vote/",
    }

    # 538 URLs
    FIVE38_BASE = "https://projects.fivethirtyeight.com"
    FIVE38_POLLS = {
        "president_approval": "/biden-approval-rating/",
        "generic_ballot": "/2024-election-forecast/",
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
        self.sources = config.polling_sources

    def is_available(self) -> bool:
        """Polling sources are publicly available."""
        return True

    def _scrape_rcp(self, poll_type: str) -> Optional[dict]:
        """Scrape RealClearPolitics for poll data."""
        if poll_type not in self.RCP_POLLS:
            return None

        url = f"{self.RCP_BASE}{self.RCP_POLLS[poll_type]}"

        try:
            response = self.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # RCP typically shows averages in tables or specific divs
            # This is a simplified scraper - actual implementation would need
            # to adapt to RCP's current HTML structure

            result = {
                "source": "rcp",
                "poll_type": poll_type,
                "url": url,
                "scraped_at": datetime.utcnow().isoformat(),
            }

            # Look for polling average tables
            tables = soup.find_all("table", class_=re.compile(r"poll|data|average", re.I))
            if tables:
                result["has_data"] = True
                # Parse table data here
            else:
                result["has_data"] = False

            return result

        except Exception as e:
            logger.error(f"RCP scrape failed for {poll_type}: {e}")
            return None

    def _scrape_538_averages(self) -> list[dict]:
        """Scrape 538 for polling averages."""
        results = []

        try:
            # 538 often has data in JSON format in script tags or API endpoints
            # Try the approval rating page
            url = f"{self.FIVE38_BASE}/biden-approval-rating/"
            response = self.get(url)

            # Look for JSON data embedded in the page
            soup = BeautifulSoup(response.text, "html.parser")

            # 538 typically embeds data in script tags with type="application/json"
            scripts = soup.find_all("script", type="application/json")
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        results.append({
                            "source": "538",
                            "data_type": "embedded_json",
                            "data": data,
                        })
                except (json.JSONDecodeError, TypeError):
                    continue

            # Also look for data attributes
            data_elements = soup.find_all(attrs={"data-json": True})
            for elem in data_elements:
                try:
                    import json
                    data = json.loads(elem.get("data-json", "{}"))
                    results.append({
                        "source": "538",
                        "data_type": "data_attribute",
                        "data": data,
                    })
                except (json.JSONDecodeError, TypeError):
                    continue

        except Exception as e:
            logger.error(f"538 scrape failed: {e}")

        return results

    def get_polling_summary(self) -> dict:
        """Get a summary of current polling data."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "sources": {},
        }

        if "rcp" in self.sources:
            rcp_data = {}
            for poll_type in self.RCP_POLLS.keys():
                result = self._scrape_rcp(poll_type)
                if result:
                    rcp_data[poll_type] = result
            summary["sources"]["rcp"] = rcp_data

        if "538" in self.sources:
            summary["sources"]["538"] = self._scrape_538_averages()

        return summary

    def crawl(self) -> list[CrawlResult]:
        """Crawl all configured polling sources."""
        results = []
        now = datetime.utcnow()

        # RCP data
        if "rcp" in self.sources:
            for poll_type in self.RCP_POLLS.keys():
                try:
                    rcp_data = self._scrape_rcp(poll_type)
                    if rcp_data:
                        result = CrawlResult(
                            source="polling_rcp",
                            category="politics",
                            data_type="poll",
                            timestamp=now,
                            data={
                                "poll_type": poll_type,
                                "source_name": "RealClearPolitics",
                                **rcp_data,
                            },
                            metadata={"url": rcp_data.get("url")},
                        )
                        results.append(result)
                except Exception as e:
                    logger.error(f"RCP crawl failed for {poll_type}: {e}")
                    self.error_count += 1

        # 538 data
        if "538" in self.sources:
            try:
                five38_data = self._scrape_538_averages()
                for item in five38_data:
                    result = CrawlResult(
                        source="polling_538",
                        category="politics",
                        data_type="poll",
                        timestamp=now,
                        data={
                            "source_name": "FiveThirtyEight",
                            **item,
                        },
                        metadata={},
                    )
                    results.append(result)
            except Exception as e:
                logger.error(f"538 crawl failed: {e}")
                self.error_count += 1

        self.last_crawl = now
        logger.info(f"Polling crawl complete: {len(results)} results")
        return results

    def search_polls(self, topic: str) -> list[dict]:
        """Search for polls on a specific topic."""
        # This would expand to search various polling sources
        results = []

        # For now, return crawl results filtered by topic
        all_results = self.crawl()
        topic_lower = topic.lower()

        for result in all_results:
            data_str = str(result.data).lower()
            if topic_lower in data_str:
                results.append(result.data)

        return results
