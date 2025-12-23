"""FRED (Federal Reserve Economic Data) API crawler."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class FredCrawler(BaseCrawler):
    """Crawler for FRED economic data API."""

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="fred",
            rate_limit=config.fred_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config
        self.api_key = config.fred_api_key
        self.base_url = config.fred_base_url
        self.series_list = config.fred_series

    def is_available(self) -> bool:
        """Check if FRED API key is configured."""
        return bool(self.api_key)

    def get_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        limit: int = 100,
        sort_order: str = "desc",
    ) -> dict:
        """Fetch series observations from FRED."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": sort_order,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        response = self.get(f"{self.base_url}/series/observations", params=params)
        return response.json()

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata about a series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        response = self.get(f"{self.base_url}/series", params=params)
        return response.json()

    def get_latest_value(self, series_id: str) -> Optional[dict]:
        """Get the most recent value for a series."""
        data = self.get_series(series_id, limit=1)
        observations = data.get("observations", [])
        if observations:
            obs = observations[0]
            return {
                "date": obs.get("date"),
                "value": float(obs.get("value")) if obs.get("value") != "." else None,
            }
        return None

    def get_series_changes(self, series_id: str, periods: int = 12) -> dict:
        """Get series with calculated changes."""
        data = self.get_series(series_id, limit=periods + 1, sort_order="desc")
        observations = data.get("observations", [])

        if len(observations) < 2:
            return {"current": None, "changes": {}}

        values = []
        for obs in observations:
            val = obs.get("value")
            if val != ".":
                values.append({
                    "date": obs.get("date"),
                    "value": float(val),
                })

        if not values:
            return {"current": None, "changes": {}}

        current = values[0]
        changes = {}

        # Calculate period-over-period changes
        if len(values) > 1:
            changes["1_period"] = {
                "absolute": current["value"] - values[1]["value"],
                "percent": ((current["value"] - values[1]["value"]) / values[1]["value"] * 100)
                if values[1]["value"] != 0 else None,
            }

        # Year-over-year if we have enough data (assuming monthly)
        if len(values) > 12:
            changes["12_period"] = {
                "absolute": current["value"] - values[12]["value"],
                "percent": ((current["value"] - values[12]["value"]) / values[12]["value"] * 100)
                if values[12]["value"] != 0 else None,
            }

        return {"current": current, "changes": changes, "history": values[:periods]}

    def crawl(self) -> list[CrawlResult]:
        """Crawl all configured FRED series."""
        results = []
        now = datetime.utcnow()

        if not self.is_available():
            logger.warning("FRED API key not configured, skipping")
            return results

        for series_config in self.series_list:
            series_id = series_config["id"]
            try:
                # Get series with change calculations
                series_data = self.get_series_changes(series_id)

                if series_data["current"]:
                    result = CrawlResult(
                        source="fred",
                        category=series_config.get("category", "economics"),
                        data_type="indicator",
                        timestamp=now,
                        data={
                            "series_id": series_id,
                            "name": series_config.get("name", series_id),
                            "current_value": series_data["current"]["value"],
                            "current_date": series_data["current"]["date"],
                            "change_1_period": series_data["changes"].get("1_period"),
                            "change_12_period": series_data["changes"].get("12_period"),
                        },
                        metadata={
                            "history": series_data.get("history", [])[:6],  # Last 6 periods
                        },
                    )
                    results.append(result)
                    logger.debug(f"FRED {series_id}: {series_data['current']['value']}")

            except Exception as e:
                logger.error(f"Failed to fetch FRED series {series_id}: {e}")
                self.error_count += 1

        self.last_crawl = now
        logger.info(f"FRED crawl complete: {len(results)} series")
        return results

    def get_economic_summary(self) -> dict:
        """Get a summary of key economic indicators."""
        results = self.crawl()

        summary = {
            "employment": {},
            "inflation": {},
            "growth": {},
            "rates": {},
        }

        for result in results:
            category = result.category
            if category in summary:
                summary[category][result.data["series_id"]] = {
                    "name": result.data["name"],
                    "value": result.data["current_value"],
                    "date": result.data["current_date"],
                    "change": result.data.get("change_1_period"),
                }

        return summary
