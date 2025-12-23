"""Betting odds crawler for election and event predictions."""

import logging
import re
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class BettingOddsCrawler(BaseCrawler):
    """Crawler for betting odds from various sportsbooks and prediction markets."""

    # Betting odds sources
    SOURCES = {
        "oddschecker": {
            "base": "https://www.oddschecker.com",
            "endpoints": {
                "politics": "/politics/us-politics",
                "president": "/politics/us-politics/us-presidential-election",
            }
        },
        "predictit_archive": {
            # PredictIt shut down but we can try to get cached data
            "base": "https://web.archive.org/web/2024",
            "endpoints": {
                "markets": "https://www.predictit.org/api/marketdata/all/",
            }
        },
        "bovada": {
            "base": "https://www.bovada.lv",
            "endpoints": {
                "politics": "/sports/politics",
            }
        },
        "betfair": {
            "base": "https://www.betfair.com",
            "endpoints": {
                "politics": "/exchange/politics",
            }
        },
    }

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="betting_odds",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config

    def is_available(self) -> bool:
        return True

    def _scrape_oddschecker(self) -> list[dict]:
        """Scrape OddsChecker for political betting odds."""
        results = []
        base = self.SOURCES["oddschecker"]["base"]

        try:
            url = f"{base}/politics/us-politics"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find betting markets
            markets = soup.find_all("div", class_=re.compile(r"market|event|bet-item", re.I))

            for market in markets[:20]:
                try:
                    result = {
                        "source": "oddschecker",
                        "scraped_at": datetime.utcnow().isoformat(),
                    }

                    # Event name
                    title = market.find(["h2", "h3", "a", "span"], class_=re.compile(r"name|title|event", re.I))
                    if title:
                        result["event"] = title.get_text(strip=True)

                    # Odds
                    odds_elems = market.find_all(class_=re.compile(r"odds|price|decimal", re.I))
                    if odds_elems:
                        result["odds"] = [o.get_text(strip=True) for o in odds_elems[:5]]

                    if result.get("event"):
                        results.append(result)

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"OddsChecker scrape failed: {e}")

        return results

    def _scrape_bovada(self) -> list[dict]:
        """Scrape Bovada for political betting lines."""
        results = []

        try:
            # Bovada has a JSON API
            url = "https://www.bovada.lv/services/sports/event/v2/events/A/description/politics"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)

            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()

                # Parse Bovada's nested structure
                events = data if isinstance(data, list) else data.get("events", [])

                for event in events[:20]:
                    try:
                        result = {
                            "source": "bovada",
                            "event_id": event.get("id"),
                            "event": event.get("description", ""),
                            "start_time": event.get("startTime"),
                            "scraped_at": datetime.utcnow().isoformat(),
                        }

                        # Extract markets/outcomes
                        display_groups = event.get("displayGroups", [])
                        for group in display_groups[:3]:
                            markets = group.get("markets", [])
                            for market in markets[:5]:
                                outcomes = market.get("outcomes", [])
                                result["outcomes"] = [
                                    {
                                        "name": o.get("description"),
                                        "price": o.get("price", {}).get("american"),
                                        "decimal": o.get("price", {}).get("decimal"),
                                    }
                                    for o in outcomes[:5]
                                ]

                        if result.get("event"):
                            results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Bovada scrape failed: {e}")

        return results

    def _get_election_betting_aggregate(self) -> list[dict]:
        """Get aggregated election betting from ElectionBettingOdds.com."""
        results = []

        try:
            url = "https://electionbettingodds.com/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find probability tables
            tables = soup.find_all("table")

            for table in tables[:5]:
                rows = table.find_all("tr")
                for row in rows[:10]:
                    try:
                        cells = row.find_all(["td", "th"])
                        if len(cells) >= 2:
                            result = {
                                "source": "electionbettingodds",
                                "scraped_at": datetime.utcnow().isoformat(),
                                "candidate": cells[0].get_text(strip=True),
                            }

                            # Look for percentage probabilities
                            for cell in cells[1:]:
                                text = cell.get_text(strip=True)
                                if "%" in text:
                                    try:
                                        prob = float(text.replace("%", "").strip())
                                        result["probability"] = prob / 100
                                        result["implied_odds"] = text
                                        break
                                    except:
                                        pass

                            if result.get("candidate") and result.get("probability"):
                                results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"ElectionBettingOdds scrape failed: {e}")

        return results

    def _scrape_metaculus(self) -> list[dict]:
        """Scrape Metaculus for prediction questions."""
        results = []

        try:
            # Metaculus has an API
            url = "https://www.metaculus.com/api2/questions/?limit=50&order_by=-activity"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                questions = data.get("results", [])

                for q in questions[:30]:
                    try:
                        # Filter for relevant questions
                        title = q.get("title", "").lower()
                        relevant_keywords = ["election", "president", "trump", "biden", "fed", "rate",
                                           "inflation", "gdp", "unemployment", "congress", "senate"]

                        if any(kw in title for kw in relevant_keywords):
                            result = {
                                "source": "metaculus",
                                "question_id": q.get("id"),
                                "title": q.get("title"),
                                "url": f"https://www.metaculus.com/questions/{q.get('id')}",
                                "community_prediction": q.get("community_prediction", {}).get("full", {}).get("q2"),
                                "num_predictions": q.get("number_of_predictions"),
                                "close_time": q.get("close_time"),
                                "resolve_time": q.get("resolve_time"),
                                "scraped_at": datetime.utcnow().isoformat(),
                            }
                            results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Metaculus scrape failed: {e}")

        return results

    def crawl(self) -> list[CrawlResult]:
        """Crawl all betting odds sources."""
        results = []
        now = datetime.utcnow()

        # OddsChecker
        try:
            for item in self._scrape_oddschecker():
                results.append(CrawlResult(
                    source="betting_oddschecker",
                    category="politics",
                    data_type="betting_odds",
                    timestamp=now,
                    data=item,
                    metadata={"betting_source": "oddschecker"},
                ))
        except Exception as e:
            logger.error(f"OddsChecker crawl failed: {e}")
            self.error_count += 1

        # Bovada
        try:
            for item in self._scrape_bovada():
                results.append(CrawlResult(
                    source="betting_bovada",
                    category="politics",
                    data_type="betting_odds",
                    timestamp=now,
                    data=item,
                    metadata={"betting_source": "bovada"},
                ))
        except Exception as e:
            logger.error(f"Bovada crawl failed: {e}")
            self.error_count += 1

        # Election Betting Odds aggregate
        try:
            for item in self._get_election_betting_aggregate():
                results.append(CrawlResult(
                    source="betting_election",
                    category="politics",
                    data_type="betting_odds",
                    timestamp=now,
                    data=item,
                    metadata={"betting_source": "electionbettingodds"},
                ))
        except Exception as e:
            logger.error(f"ElectionBettingOdds crawl failed: {e}")
            self.error_count += 1

        # Metaculus predictions
        try:
            for item in self._scrape_metaculus():
                results.append(CrawlResult(
                    source="betting_metaculus",
                    category="predictions",
                    data_type="betting_odds",
                    timestamp=now,
                    data=item,
                    metadata={"betting_source": "metaculus"},
                ))
        except Exception as e:
            logger.error(f"Metaculus crawl failed: {e}")
            self.error_count += 1

        self.last_crawl = now
        logger.info(f"Betting odds crawl complete: {len(results)} results")
        return results

    def get_election_consensus(self) -> dict:
        """Get consensus probabilities across betting markets."""
        results = self.crawl()

        # Aggregate by candidate
        candidates = {}
        for result in results:
            if result.data_type == "betting_odds":
                candidate = result.data.get("candidate") or result.data.get("event", "")
                prob = result.data.get("probability")

                if candidate and prob:
                    if candidate not in candidates:
                        candidates[candidate] = []
                    candidates[candidate].append(prob)

        # Calculate averages
        consensus = {}
        for candidate, probs in candidates.items():
            consensus[candidate] = {
                "avg_probability": sum(probs) / len(probs),
                "num_sources": len(probs),
                "range": [min(probs), max(probs)],
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "candidates": consensus,
        }
