"""Economic calendar crawler for FOMC meetings, BLS releases, etc."""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class EconomicCalendarCrawler(BaseCrawler):
    """Crawler for economic event calendar data."""

    # Key economic events to track
    TRACKED_EVENTS = [
        "FOMC",
        "Fed",
        "CPI",
        "PPI",
        "NFP",
        "Nonfarm",
        "Payroll",
        "GDP",
        "Unemployment",
        "Retail Sales",
        "Consumer Confidence",
        "ISM",
        "Housing",
        "Durable Goods",
        "Trade Balance",
        "PCE",
        "Core PCE",
    ]

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="economic_calendar",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config

    def is_available(self) -> bool:
        return True

    def _scrape_investing_calendar(self) -> list[dict]:
        """Scrape economic calendar from Investing.com."""
        events = []

        try:
            url = "https://www.investing.com/economic-calendar/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find event rows
            rows = soup.find_all("tr", class_=re.compile(r"js-event-item"))

            for row in rows[:50]:  # First 50 events
                try:
                    event = {}

                    # Time
                    time_cell = row.find("td", class_="time")
                    if time_cell:
                        event["time"] = time_cell.get_text(strip=True)

                    # Country
                    flag = row.find("td", class_="flagCur")
                    if flag:
                        event["country"] = flag.get_text(strip=True)

                    # Event name
                    event_cell = row.find("td", class_="event")
                    if event_cell:
                        event["name"] = event_cell.get_text(strip=True)

                    # Impact (bulls)
                    impact = row.find("td", class_="sentiment")
                    if impact:
                        bulls = impact.find_all("i", class_="grayFullBullishIcon")
                        event["impact"] = len(bulls)  # 1-3 bulls = low-high impact

                    # Actual/Forecast/Previous
                    actual = row.find("td", class_="act")
                    forecast = row.find("td", class_="fore")
                    previous = row.find("td", class_="prev")

                    if actual:
                        event["actual"] = actual.get_text(strip=True)
                    if forecast:
                        event["forecast"] = forecast.get_text(strip=True)
                    if previous:
                        event["previous"] = previous.get_text(strip=True)

                    if event.get("name"):
                        events.append(event)

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Investing.com calendar scrape failed: {e}")

        return events

    def _scrape_fed_calendar(self) -> list[dict]:
        """Scrape Federal Reserve calendar."""
        events = []

        try:
            url = "https://www.federalreserve.gov/newsevents/calendar.htm"
            response = self.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find event items
            items = soup.find_all("div", class_="row")

            for item in items[:20]:
                try:
                    date_elem = item.find("time")
                    title_elem = item.find("em") or item.find("a")

                    if date_elem and title_elem:
                        events.append({
                            "source": "fed",
                            "date": date_elem.get_text(strip=True),
                            "name": title_elem.get_text(strip=True),
                            "type": "fed_event",
                        })
                except:
                    continue

        except Exception as e:
            logger.error(f"Fed calendar scrape failed: {e}")

        return events

    def _get_upcoming_fomc(self) -> list[dict]:
        """Get upcoming FOMC meeting dates (hardcoded for 2024-2025)."""
        # FOMC meeting dates (update annually)
        fomc_dates = [
            # 2024 remaining
            "2024-12-17",
            "2024-12-18",
            # 2025
            "2025-01-28",
            "2025-01-29",
            "2025-03-18",
            "2025-03-19",
            "2025-05-06",
            "2025-05-07",
            "2025-06-17",
            "2025-06-18",
            "2025-07-29",
            "2025-07-30",
            "2025-09-16",
            "2025-09-17",
            "2025-11-04",
            "2025-11-05",
            "2025-12-16",
            "2025-12-17",
        ]

        events = []
        today = datetime.now().date()

        for date_str in fomc_dates:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if meeting_date >= today:
                days_until = (meeting_date - today).days
                events.append({
                    "source": "fomc",
                    "date": date_str,
                    "name": "FOMC Meeting",
                    "days_until": days_until,
                    "type": "fomc",
                    "impact": 3,  # High impact
                })

        return events[:6]  # Next 6 meetings

    def _get_bls_schedule(self) -> list[dict]:
        """Get upcoming BLS release dates."""
        # Key BLS releases happen on specific patterns
        # CPI: Usually 2nd or 3rd week of month
        # Jobs: First Friday of month
        events = []
        today = datetime.now().date()

        # Generate approximate dates for next 3 months
        for month_offset in range(4):
            target_month = today.month + month_offset
            target_year = today.year

            if target_month > 12:
                target_month -= 12
                target_year += 1

            # Jobs report: First Friday
            first_day = datetime(target_year, target_month, 1).date()
            days_until_friday = (4 - first_day.weekday()) % 7
            jobs_date = first_day + timedelta(days=days_until_friday)

            if jobs_date >= today:
                events.append({
                    "source": "bls",
                    "date": jobs_date.isoformat(),
                    "name": "Employment Situation (Jobs Report)",
                    "days_until": (jobs_date - today).days,
                    "type": "jobs_report",
                    "impact": 3,
                })

            # CPI: Usually around 13th of month
            cpi_date = datetime(target_year, target_month, 13).date()
            if cpi_date >= today:
                events.append({
                    "source": "bls",
                    "date": cpi_date.isoformat(),
                    "name": "Consumer Price Index (CPI)",
                    "days_until": (cpi_date - today).days,
                    "type": "cpi",
                    "impact": 3,
                })

        return sorted(events, key=lambda x: x["days_until"])[:10]

    def crawl(self) -> list[CrawlResult]:
        """Crawl all economic calendar sources."""
        results = []
        now = datetime.utcnow()

        # FOMC meetings
        try:
            fomc_events = self._get_upcoming_fomc()
            for event in fomc_events:
                results.append(CrawlResult(
                    source="calendar_fomc",
                    category="economics",
                    data_type="calendar",
                    timestamp=now,
                    data=event,
                    metadata={"event_type": "fomc"},
                ))
        except Exception as e:
            logger.error(f"FOMC calendar failed: {e}")

        # BLS releases
        try:
            bls_events = self._get_bls_schedule()
            for event in bls_events:
                results.append(CrawlResult(
                    source="calendar_bls",
                    category="economics",
                    data_type="calendar",
                    timestamp=now,
                    data=event,
                    metadata={"event_type": "bls"},
                ))
        except Exception as e:
            logger.error(f"BLS calendar failed: {e}")

        # Investing.com calendar (more comprehensive)
        try:
            investing_events = self._scrape_investing_calendar()
            for event in investing_events:
                # Filter to US and high impact
                if event.get("country", "").strip() in ["USD", "US", ""]:
                    if event.get("impact", 0) >= 2:  # Medium+ impact
                        results.append(CrawlResult(
                            source="calendar_investing",
                            category="economics",
                            data_type="calendar",
                            timestamp=now,
                            data=event,
                            metadata={"event_type": "general"},
                        ))
        except Exception as e:
            logger.error(f"Investing.com calendar failed: {e}")

        self.last_crawl = now
        logger.info(f"Economic calendar crawl complete: {len(results)} events")
        return results

    def get_upcoming_events(self, days: int = 7) -> list[dict]:
        """Get events happening in the next N days."""
        results = self.crawl()
        upcoming = []

        for result in results:
            days_until = result.data.get("days_until")
            if days_until is not None and days_until <= days:
                upcoming.append(result.data)

        return sorted(upcoming, key=lambda x: x.get("days_until", 999))
