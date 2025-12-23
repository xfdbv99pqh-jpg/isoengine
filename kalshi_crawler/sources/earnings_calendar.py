"""Earnings calendar crawler for major company reports."""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class EarningsCalendarCrawler(BaseCrawler):
    """Crawler for upcoming earnings reports from major companies."""

    # Key companies to track (relevant to prediction markets)
    TRACKED_COMPANIES = {
        # Mega-cap tech
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "META": "Meta",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        # Other notable
        "NFLX": "Netflix",
        "AMD": "AMD",
        "INTC": "Intel",
        "CRM": "Salesforce",
        "ORCL": "Oracle",
        "IBM": "IBM",
        "UBER": "Uber",
        "ABNB": "Airbnb",
        "COIN": "Coinbase",
        "SQ": "Block",
        "PYPL": "PayPal",
        # Banks/Finance
        "JPM": "JPMorgan",
        "BAC": "Bank of America",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        # Energy
        "XOM": "Exxon",
        "CVX": "Chevron",
        # Retail
        "WMT": "Walmart",
        "TGT": "Target",
        "COST": "Costco",
    }

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="earnings_calendar",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config

    def is_available(self) -> bool:
        return True

    def _scrape_yahoo_earnings(self) -> list[dict]:
        """Scrape Yahoo Finance earnings calendar."""
        results = []

        try:
            # Yahoo earnings calendar
            today = datetime.now()
            dates_to_check = [today + timedelta(days=i) for i in range(14)]

            for date in dates_to_check:
                date_str = date.strftime("%Y-%m-%d")
                url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }

                try:
                    response = self.session.get(url, headers=headers, timeout=self.timeout)
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Find earnings table
                    table = soup.find("table")
                    if not table:
                        continue

                    rows = table.find_all("tr")[1:]  # Skip header

                    for row in rows[:50]:
                        try:
                            cells = row.find_all("td")
                            if len(cells) >= 4:
                                symbol = cells[0].get_text(strip=True)
                                company = cells[1].get_text(strip=True)

                                result = {
                                    "source": "yahoo",
                                    "symbol": symbol,
                                    "company": company,
                                    "earnings_date": date_str,
                                    "days_until": (date - today).days,
                                    "scraped_at": datetime.utcnow().isoformat(),
                                }

                                # EPS estimates if available
                                if len(cells) >= 5:
                                    result["eps_estimate"] = cells[2].get_text(strip=True)
                                    result["eps_actual"] = cells[3].get_text(strip=True)

                                # Track if it's a key company
                                result["is_tracked"] = symbol in self.TRACKED_COMPANIES

                                results.append(result)

                        except Exception:
                            continue

                except Exception as e:
                    logger.debug(f"Yahoo earnings for {date_str} failed: {e}")
                    continue

        except Exception as e:
            logger.error(f"Yahoo earnings scrape failed: {e}")

        return results

    def _scrape_nasdaq_earnings(self) -> list[dict]:
        """Scrape NASDAQ earnings calendar."""
        results = []

        try:
            # NASDAQ earnings calendar API
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"https://api.nasdaq.com/api/calendar/earnings?date={today}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                rows = data.get("data", {}).get("rows", [])

                for row in rows[:50]:
                    try:
                        result = {
                            "source": "nasdaq",
                            "symbol": row.get("symbol"),
                            "company": row.get("name"),
                            "earnings_date": row.get("date"),
                            "time": row.get("time"),  # Before/After market
                            "eps_forecast": row.get("epsForecast"),
                            "num_estimates": row.get("noOfEsts"),
                            "scraped_at": datetime.utcnow().isoformat(),
                        }

                        # Track if it's a key company
                        symbol = row.get("symbol", "")
                        result["is_tracked"] = symbol in self.TRACKED_COMPANIES

                        results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"NASDAQ earnings scrape failed: {e}")

        return results

    def _scrape_earningswhispers(self) -> list[dict]:
        """Scrape Earnings Whispers for calendar and estimates."""
        results = []

        try:
            url = "https://www.earningswhispers.com/calendar"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find earnings entries
            entries = soup.find_all(class_=re.compile(r"ticker|earning|company", re.I))

            for entry in entries[:30]:
                try:
                    result = {
                        "source": "earningswhispers",
                        "scraped_at": datetime.utcnow().isoformat(),
                    }

                    # Symbol
                    symbol_elem = entry.find(class_=re.compile(r"symbol|ticker", re.I))
                    if symbol_elem:
                        result["symbol"] = symbol_elem.get_text(strip=True)

                    # Date
                    date_elem = entry.find(class_=re.compile(r"date|time", re.I))
                    if date_elem:
                        result["earnings_date"] = date_elem.get_text(strip=True)

                    if result.get("symbol"):
                        result["is_tracked"] = result["symbol"] in self.TRACKED_COMPANIES
                        results.append(result)

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"EarningsWhispers scrape failed: {e}")

        return results

    def _get_key_earnings_dates(self) -> list[dict]:
        """Generate known upcoming earnings dates for major companies."""
        # Earnings seasons: mid-Jan, mid-Apr, mid-Jul, mid-Oct
        results = []
        today = datetime.now().date()

        # Approximate earnings timing based on fiscal quarters
        earnings_seasons = [
            (1, 20),   # Q4 earnings in late January
            (4, 20),   # Q1 earnings in late April
            (7, 20),   # Q2 earnings in late July
            (10, 20),  # Q3 earnings in late October
        ]

        for symbol, company in self.TRACKED_COMPANIES.items():
            # Find next earnings window
            for month, day in earnings_seasons:
                year = today.year if (month, day) >= (today.month, today.day) else today.year + 1
                earnings_date = datetime(year, month, day).date()

                if earnings_date >= today and (earnings_date - today).days <= 90:
                    results.append({
                        "source": "estimated",
                        "symbol": symbol,
                        "company": company,
                        "earnings_date": earnings_date.isoformat(),
                        "days_until": (earnings_date - today).days,
                        "is_estimate": True,
                        "is_tracked": True,
                        "scraped_at": datetime.utcnow().isoformat(),
                    })
                    break  # Only next earnings

        return sorted(results, key=lambda x: x.get("days_until", 999))

    def crawl(self) -> list[CrawlResult]:
        """Crawl all earnings calendar sources."""
        results = []
        now = datetime.utcnow()

        # Yahoo Finance
        try:
            for item in self._scrape_yahoo_earnings():
                results.append(CrawlResult(
                    source="earnings_yahoo",
                    category="earnings",
                    data_type="calendar",
                    timestamp=now,
                    data=item,
                    metadata={"symbol": item.get("symbol")},
                ))
        except Exception as e:
            logger.error(f"Yahoo earnings crawl failed: {e}")
            self.error_count += 1

        # NASDAQ
        try:
            for item in self._scrape_nasdaq_earnings():
                results.append(CrawlResult(
                    source="earnings_nasdaq",
                    category="earnings",
                    data_type="calendar",
                    timestamp=now,
                    data=item,
                    metadata={"symbol": item.get("symbol")},
                ))
        except Exception as e:
            logger.error(f"NASDAQ earnings crawl failed: {e}")
            self.error_count += 1

        # Earnings Whispers
        try:
            for item in self._scrape_earningswhispers():
                results.append(CrawlResult(
                    source="earnings_whispers",
                    category="earnings",
                    data_type="calendar",
                    timestamp=now,
                    data=item,
                    metadata={"symbol": item.get("symbol")},
                ))
        except Exception as e:
            logger.error(f"EarningsWhispers crawl failed: {e}")
            self.error_count += 1

        # Known key company dates (as fallback/supplement)
        try:
            for item in self._get_key_earnings_dates():
                results.append(CrawlResult(
                    source="earnings_estimated",
                    category="earnings",
                    data_type="calendar",
                    timestamp=now,
                    data=item,
                    metadata={"symbol": item.get("symbol"), "estimated": True},
                ))
        except Exception as e:
            logger.error(f"Key earnings dates failed: {e}")
            self.error_count += 1

        self.last_crawl = now
        logger.info(f"Earnings calendar crawl complete: {len(results)} results")
        return results

    def get_upcoming_earnings(self, days: int = 7) -> list[dict]:
        """Get earnings happening in the next N days."""
        results = self.crawl()
        upcoming = []

        for result in results:
            days_until = result.data.get("days_until")
            if days_until is not None and days_until <= days:
                upcoming.append(result.data)

        # Deduplicate by symbol
        seen = set()
        deduped = []
        for item in sorted(upcoming, key=lambda x: x.get("days_until", 999)):
            symbol = item.get("symbol")
            if symbol and symbol not in seen:
                seen.add(symbol)
                deduped.append(item)

        return deduped

    def get_tracked_company_earnings(self) -> list[dict]:
        """Get earnings for tracked major companies."""
        results = self.crawl()
        return [
            r.data for r in results
            if r.data.get("is_tracked")
        ]
