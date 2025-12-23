"""Signal extraction and cross-market comparison."""

import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from ..storage.db import CrawlerDatabase
from ..base import CrawlResult

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal or alert."""

    signal_type: str
    source: str
    ticker: Optional[str]
    message: str
    severity: str  # "info", "warning", "alert"
    data: dict
    timestamp: datetime

    def __str__(self):
        return f"[{self.severity.upper()}] {self.signal_type}: {self.message}"


class SignalAnalyzer:
    """Analyzes crawl data to generate trading signals."""

    def __init__(self, db: CrawlerDatabase):
        self.db = db

    def analyze_all(self) -> list[Signal]:
        """Run all signal analyses."""
        signals = []

        signals.extend(self.detect_price_movements())
        signals.extend(self.detect_cross_market_arbitrage())
        signals.extend(self.detect_indicator_surprises())
        signals.extend(self.detect_news_keywords())

        # Store signals
        for signal in signals:
            self.db.store_signal(
                signal_type=signal.signal_type,
                message=signal.message,
                source=signal.source,
                ticker=signal.ticker,
                severity=signal.severity,
                data=signal.data,
            )

        return signals

    def detect_price_movements(self, threshold: float = 0.10) -> list[Signal]:
        """Detect significant price movements in markets."""
        signals = []

        markets = self.db.get_latest_markets(limit=500)

        for market in markets:
            ticker = market.get("ticker")
            if not ticker:
                continue

            # Get price history
            history = self.db.get_price_history(ticker, hours=24)
            if len(history) < 2:
                continue

            current_price = history[-1].get("price")
            oldest_price = history[0].get("price")

            if current_price is None or oldest_price is None or oldest_price == 0:
                continue

            change = (current_price - oldest_price) / oldest_price

            if abs(change) >= threshold:
                direction = "up" if change > 0 else "down"
                severity = "alert" if abs(change) >= 0.20 else "warning"

                signals.append(Signal(
                    signal_type="price_movement",
                    source=market.get("source", "unknown"),
                    ticker=ticker,
                    message=f"{ticker} moved {direction} {abs(change)*100:.1f}% in 24h (${oldest_price:.2f} -> ${current_price:.2f})",
                    severity=severity,
                    data={
                        "current_price": current_price,
                        "previous_price": oldest_price,
                        "change_percent": change * 100,
                        "title": market.get("title"),
                    },
                    timestamp=datetime.utcnow(),
                ))

        return signals

    def detect_cross_market_arbitrage(self, threshold: float = 0.05) -> list[Signal]:
        """Detect price discrepancies between Kalshi and Polymarket."""
        signals = []

        kalshi_markets = self.db.get_latest_markets(source="kalshi")
        polymarket_markets = self.db.get_latest_markets(source="polymarket")

        # Build lookup for Polymarket
        poly_by_keywords = {}
        for market in polymarket_markets:
            import json
            data = json.loads(market.get("data", "{}"))
            question = data.get("question", "").lower()
            # Extract keywords
            words = [w for w in question.split() if len(w) > 4][:5]
            for word in words:
                if word not in poly_by_keywords:
                    poly_by_keywords[word] = []
                poly_by_keywords[word].append(market)

        # Compare Kalshi markets with potential Polymarket matches
        for kalshi in kalshi_markets:
            import json
            k_data = json.loads(kalshi.get("data", "{}"))
            k_title = k_data.get("title", "").lower()
            k_price = k_data.get("yes_price")

            if k_price is None:
                continue

            # Find potential matches
            title_words = [w for w in k_title.split() if len(w) > 4][:5]
            candidates = set()
            for word in title_words:
                for match in poly_by_keywords.get(word, []):
                    candidates.add(match.get("id"))

            # Check for significant price differences
            for poly in polymarket_markets:
                if poly.get("id") not in candidates:
                    continue

                p_data = json.loads(poly.get("data", "{}"))
                prices = p_data.get("prices", {})
                p_price = prices.get("Yes") or prices.get("yes")

                if p_price is None:
                    continue

                diff = abs(k_price - p_price)
                if diff >= threshold:
                    higher = "Kalshi" if k_price > p_price else "Polymarket"
                    signals.append(Signal(
                        signal_type="cross_market_arb",
                        source="comparison",
                        ticker=kalshi.get("ticker"),
                        message=f"Price gap: {higher} is {diff*100:.1f}% higher for similar markets",
                        severity="alert" if diff >= 0.10 else "warning",
                        data={
                            "kalshi_ticker": kalshi.get("ticker"),
                            "kalshi_title": k_data.get("title"),
                            "kalshi_price": k_price,
                            "polymarket_question": p_data.get("question"),
                            "polymarket_price": p_price,
                            "difference": diff,
                        },
                        timestamp=datetime.utcnow(),
                    ))

        return signals

    def detect_indicator_surprises(self) -> list[Signal]:
        """Detect when economic indicators show surprising changes."""
        signals = []

        indicators = self.db.get_latest_indicators()

        for indicator in indicators:
            import json
            data = json.loads(indicator.get("data", "{}"))

            change = data.get("change_1_period", {})
            if not change:
                continue

            percent_change = change.get("percent")
            if percent_change is None:
                continue

            series_id = indicator.get("series_id")
            name = indicator.get("name") or series_id

            # Define thresholds by type
            thresholds = {
                "UNRATE": 0.1,  # Unemployment: 0.1 percentage point
                "CPIAUCSL": 0.5,  # CPI: 0.5% monthly change
                "FEDFUNDS": 0.25,  # Fed funds: 25bps
                "DGS10": 0.10,  # 10Y yield: 10bps
            }

            threshold = thresholds.get(series_id, 5.0)  # Default 5% change

            if abs(percent_change) >= threshold:
                direction = "increased" if percent_change > 0 else "decreased"
                signals.append(Signal(
                    signal_type="indicator_surprise",
                    source="fred",
                    ticker=series_id,
                    message=f"{name} {direction} {abs(percent_change):.2f}% (threshold: {threshold}%)",
                    severity="warning",
                    data={
                        "series_id": series_id,
                        "name": name,
                        "current_value": data.get("current_value"),
                        "percent_change": percent_change,
                        "threshold": threshold,
                    },
                    timestamp=datetime.utcnow(),
                ))

        return signals

    def detect_news_keywords(self) -> list[Signal]:
        """Detect important keywords in recent news."""
        signals = []

        # Keywords that might signal market-moving events
        alert_keywords = {
            "breaking": "alert",
            "emergency": "alert",
            "recession": "warning",
            "crash": "warning",
            "surge": "info",
            "plunge": "warning",
            "fed rate": "info",
            "inflation data": "info",
            "jobs report": "info",
            "election results": "info",
        }

        news = self.db.get_recent_news(hours=6)

        for item in news:
            title = (item.get("title") or "").lower()
            content = (item.get("content") or "").lower()
            text = f"{title} {content}"

            for keyword, severity in alert_keywords.items():
                if keyword in text:
                    signals.append(Signal(
                        signal_type="news_alert",
                        source=item.get("source", "news"),
                        ticker=None,
                        message=f"Keyword '{keyword}' detected: {item.get('title', '')[:100]}",
                        severity=severity,
                        data={
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "keyword": keyword,
                            "category": item.get("category"),
                        },
                        timestamp=datetime.utcnow(),
                    ))
                    break  # One signal per news item

        return signals

    def compare_kalshi_to_indicators(self) -> list[dict]:
        """Compare Kalshi market prices to underlying indicators."""
        comparisons = []

        # Map market types to indicators
        market_indicator_map = {
            "CPI": ["CPIAUCSL", "CPILFESL"],
            "UNEMPLOYMENT": ["UNRATE"],
            "GDP": ["GDP", "GDPC1"],
            "FED": ["FEDFUNDS"],
        }

        kalshi_markets = self.db.get_latest_markets(source="kalshi")
        indicators = {i["series_id"]: i for i in self.db.get_latest_indicators()}

        for market in kalshi_markets:
            import json
            data = json.loads(market.get("data", "{}"))
            ticker = market.get("ticker", "").upper()

            # Find matching indicators
            matching_indicators = []
            for prefix, series_ids in market_indicator_map.items():
                if prefix in ticker:
                    for sid in series_ids:
                        if sid in indicators:
                            matching_indicators.append(indicators[sid])

            if matching_indicators:
                comparisons.append({
                    "market": {
                        "ticker": ticker,
                        "title": data.get("title"),
                        "yes_price": data.get("yes_price"),
                    },
                    "indicators": [
                        {
                            "series_id": ind["series_id"],
                            "name": ind.get("name"),
                            "value": ind.get("value"),
                            "date": ind.get("value_date"),
                        }
                        for ind in matching_indicators
                    ],
                })

        return comparisons

    def get_market_summary(self) -> dict:
        """Get a summary of current market state."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "kalshi_markets": len(self.db.get_latest_markets(source="kalshi")),
            "polymarket_markets": len(self.db.get_latest_markets(source="polymarket")),
            "indicators": len(self.db.get_latest_indicators()),
            "recent_news": len(self.db.get_recent_news(hours=24)),
            "unacknowledged_signals": len(self.db.get_unacknowledged_signals()),
            "crawl_stats": self.db.get_crawl_stats(hours=24),
        }
