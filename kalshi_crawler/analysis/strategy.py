"""
Kalshi Investment Strategy Generator

Sophisticated analysis combining:
- Price momentum and trends
- Cross-market arbitrage (Kalshi vs Polymarket)
- Economic indicator alignment
- Volume and liquidity analysis
- Expiration timing
- News sentiment
"""

import json
import logging
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from ..storage.db import CrawlerDatabase

logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    """A recommended trade with reasoning."""

    ticker: str
    title: str
    action: str  # "BUY_YES", "BUY_NO", "HOLD", "AVOID"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    current_price: Optional[float]
    target_price: Optional[float]
    reasoning: list[str]
    category: str
    data_sources: list[str]
    timestamp: datetime
    edge: Optional[float] = None  # Expected edge/alpha

    def __str__(self):
        arrow = "↑" if self.action == "BUY_YES" else "↓" if self.action == "BUY_NO" else "→"
        price_str = f"${self.current_price:.2f}" if self.current_price else "N/A"
        edge_str = f" (edge: {self.edge:+.1f}%)" if self.edge else ""
        return f"[{self.confidence}] {arrow} {self.action} {self.ticker} @ {price_str}{edge_str}\n    {self.title[:60]}..."

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "title": self.title,
            "action": self.action,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "reasoning": self.reasoning,
            "category": self.category,
            "data_sources": self.data_sources,
            "timestamp": self.timestamp.isoformat(),
            "edge": self.edge,
        }


class StrategyGenerator:
    """Generates Kalshi trading strategies based on collected data."""

    def __init__(self, db: CrawlerDatabase):
        self.db = db

    def generate_recommendations(self) -> list[TradeRecommendation]:
        """Generate all trading recommendations."""
        recommendations = []

        # 1. Cross-market arbitrage (highest confidence)
        recommendations.extend(self._find_arbitrage_opportunities())

        # 2. Betting odds arbitrage (cross-platform)
        recommendations.extend(self._find_betting_odds_arbitrage())

        # 3. Economic indicator misalignment
        recommendations.extend(self._analyze_economic_misalignment())

        # 4. Calendar-driven opportunities (FOMC, BLS, earnings)
        recommendations.extend(self._analyze_calendar_events())

        # 5. High volume + price extreme opportunities
        recommendations.extend(self._analyze_volume_price_extremes())

        # 6. Expiring soon with uncertain prices
        recommendations.extend(self._analyze_expiring_markets())

        # 7. News-driven momentum
        recommendations.extend(self._analyze_news_momentum())

        # 8. Social sentiment signals
        recommendations.extend(self._analyze_social_sentiment())

        # Sort by confidence then edge
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: (
            confidence_order.get(x.confidence, 3),
            -(x.edge or 0)
        ))

        return recommendations

    def _find_arbitrage_opportunities(self) -> list[TradeRecommendation]:
        """Find price discrepancies between Kalshi and Polymarket."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)
        poly_markets = self.db.get_latest_markets(source="polymarket", limit=500)

        # Build searchable index of Polymarket markets
        poly_index = []
        for pm in poly_markets:
            data = json.loads(pm.get("data", "{}"))
            question = (data.get("question") or "").lower()
            prices = data.get("prices", {})
            yes_price = prices.get("Yes") or prices.get("yes")
            if yes_price and question:
                poly_index.append({
                    "question": question,
                    "price": yes_price,
                    "data": data,
                    "words": set(re.findall(r'\b\w{4,}\b', question)),
                })

        # Compare each Kalshi market
        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            k_title = (k_data.get("title") or "").lower()
            k_price = k_data.get("yes_price")
            k_ticker = k_data.get("ticker", "UNKNOWN")

            if not k_price or not k_title:
                continue

            k_words = set(re.findall(r'\b\w{4,}\b', k_title))

            # Find matching Polymarket markets
            for pm in poly_index:
                # Calculate word overlap
                common = k_words & pm["words"]
                if len(common) < 3:
                    continue

                # Check for significant price difference
                p_price = pm["price"]
                diff = abs(k_price - p_price)

                if diff >= 0.05:  # 5%+ difference
                    edge = diff * 100
                    higher = "Kalshi" if k_price > p_price else "Polymarket"
                    lower = "Polymarket" if k_price > p_price else "Kalshi"

                    # Determine action
                    if k_price > p_price:
                        action = "BUY_NO"  # Kalshi overpriced, buy NO
                        reasoning = [
                            f"Kalshi YES @ ${k_price:.2f} vs Polymarket @ ${p_price:.2f}",
                            f"Kalshi is {diff*100:.1f}% higher - potential overpricing",
                            f"Matching market: {pm['data'].get('question', '')[:50]}...",
                        ]
                    else:
                        action = "BUY_YES"  # Kalshi underpriced, buy YES
                        reasoning = [
                            f"Kalshi YES @ ${k_price:.2f} vs Polymarket @ ${p_price:.2f}",
                            f"Kalshi is {diff*100:.1f}% lower - potential value",
                            f"Matching market: {pm['data'].get('question', '')[:50]}...",
                        ]

                    confidence = "HIGH" if diff >= 0.10 else "MEDIUM"

                    recommendations.append(TradeRecommendation(
                        ticker=k_ticker,
                        title=k_data.get("title", ""),
                        action=action,
                        confidence=confidence,
                        current_price=k_price,
                        target_price=p_price,
                        reasoning=reasoning,
                        category="arbitrage",
                        data_sources=["kalshi", "polymarket"],
                        timestamp=now,
                        edge=edge,
                    ))
                    break  # One match per Kalshi market

        return recommendations[:10]  # Top 10 arb opportunities

    def _analyze_economic_misalignment(self) -> list[TradeRecommendation]:
        """Find markets where price doesn't align with economic data."""
        recommendations = []
        now = datetime.utcnow()

        # Get indicators
        indicators = {}
        for ind in self.db.get_latest_indicators():
            indicators[ind["series_id"]] = ind

        if not indicators:
            return recommendations

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        # Key economic thresholds
        fed_rate = None
        if "FEDFUNDS" in indicators:
            ff_data = json.loads(indicators["FEDFUNDS"].get("data", "{}"))
            fed_rate = ff_data.get("current_value")

        unemployment = None
        if "UNRATE" in indicators:
            ur_data = json.loads(indicators["UNRATE"].get("data", "{}"))
            unemployment = ur_data.get("current_value")

        cpi_yoy = None
        if "CPIAUCSL" in indicators:
            cpi_data = json.loads(indicators["CPIAUCSL"].get("data", "{}"))
            change_data = cpi_data.get("change_12_period")
            if change_data and isinstance(change_data, dict):
                cpi_yoy = change_data.get("percent")

        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            title = (k_data.get("title") or "").lower()
            ticker = k_data.get("ticker", "UNKNOWN")
            price = k_data.get("yes_price")

            if not price:
                continue

            reasoning = []
            action = "HOLD"
            confidence = "LOW"
            edge = None

            # Fed rate markets
            if fed_rate and any(kw in title for kw in ["rate cut", "rate hike", "fomc", "federal reserve"]):
                # Extract target rate from title if possible
                rate_match = re.search(r'(\d+\.?\d*)%', title)
                if rate_match:
                    target_rate = float(rate_match.group(1))

                    if "cut" in title and fed_rate > target_rate:
                        # Market expects cut, current rate is higher
                        if price < 0.40:
                            reasoning.append(f"Fed rate at {fed_rate:.2f}%, market sees cut to {target_rate}% as unlikely ({price*100:.0f}%)")
                            reasoning.append("Fed has been cutting - momentum toward lower rates")
                            action = "BUY_YES"
                            confidence = "MEDIUM"
                            edge = (0.50 - price) * 100

            # Inflation markets
            if cpi_yoy is not None and any(kw in title for kw in ["inflation", "cpi"]):
                # Extract threshold from title
                pct_match = re.search(r'(\d+\.?\d*)%', title)
                if pct_match:
                    threshold = float(pct_match.group(1))

                    if "above" in title or "over" in title or "higher" in title:
                        if cpi_yoy > threshold and price < 0.60:
                            reasoning.append(f"CPI YoY at {cpi_yoy:.1f}%, already above {threshold}%")
                            reasoning.append(f"Market only pricing {price*100:.0f}% chance - undervalued?")
                            action = "BUY_YES"
                            confidence = "MEDIUM"
                            edge = (0.70 - price) * 100
                        elif cpi_yoy < threshold and price > 0.50:
                            reasoning.append(f"CPI YoY at {cpi_yoy:.1f}%, below {threshold}%")
                            reasoning.append(f"Market pricing {price*100:.0f}% - overvalued?")
                            action = "BUY_NO"
                            confidence = "MEDIUM"
                            edge = (price - 0.30) * 100

                    elif "below" in title or "under" in title or "lower" in title:
                        if cpi_yoy < threshold and price < 0.60:
                            reasoning.append(f"CPI YoY at {cpi_yoy:.1f}%, already below {threshold}%")
                            reasoning.append(f"Market only pricing {price*100:.0f}% chance - undervalued?")
                            action = "BUY_YES"
                            confidence = "MEDIUM"
                            edge = (0.70 - price) * 100

            # Unemployment markets
            if unemployment is not None and any(kw in title for kw in ["unemployment", "jobless"]):
                pct_match = re.search(r'(\d+\.?\d*)%', title)
                if pct_match:
                    threshold = float(pct_match.group(1))

                    if "above" in title or "over" in title:
                        if unemployment > threshold and price < 0.50:
                            reasoning.append(f"Unemployment at {unemployment:.1f}%, already above {threshold}%")
                            action = "BUY_YES"
                            confidence = "MEDIUM"
                            edge = (0.60 - price) * 100

            if reasoning and action != "HOLD":
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    title=k_data.get("title", ""),
                    action=action,
                    confidence=confidence,
                    current_price=price,
                    target_price=None,
                    reasoning=reasoning,
                    category="economic_data",
                    data_sources=["kalshi", "fred"],
                    timestamp=now,
                    edge=edge,
                ))

        return recommendations[:10]

    def _analyze_volume_price_extremes(self) -> list[TradeRecommendation]:
        """Find liquid markets with interesting price levels."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            ticker = k_data.get("ticker", "UNKNOWN")
            title = k_data.get("title", "")
            price = k_data.get("yes_price")
            volume = k_data.get("volume") or 0
            volume_24h = k_data.get("volume_24h") or 0
            open_interest = k_data.get("open_interest") or 0

            if not price:
                continue

            # Skip dead markets ($0.01 or $0.99) - these are essentially resolved
            if price <= 0.02 or price >= 0.98:
                continue

            # Require RECENT activity, not just historical volume
            has_recent_activity = volume_24h >= 100
            is_active = volume_24h >= 1000
            is_very_active = volume_24h >= 5000

            if not has_recent_activity:
                continue

            reasoning = []
            action = "HOLD"
            confidence = "LOW"
            edge = None

            # Moderately low price (5-15%) with recent activity - potential value
            if 0.05 <= price <= 0.15 and is_active:
                reasoning.append(f"Price ${price:.2f} - market says {price*100:.0f}% chance")
                reasoning.append(f"Active trading: {volume_24h:,} contracts in 24h")
                reasoning.append("Low price with activity suggests potential opportunity")
                action = "BUY_YES"
                confidence = "LOW"
                edge = (0.25 - price) * 100

            # Moderately high price (85-95%) with recent activity
            elif 0.85 <= price <= 0.95 and is_active:
                reasoning.append(f"Price ${price:.2f} - market says {price*100:.0f}% likely")
                reasoning.append(f"Active trading: {volume_24h:,} contracts in 24h")
                reasoning.append("High price but not certain - potential NO value")
                action = "BUY_NO"
                confidence = "LOW"
                edge = (price - 0.75) * 100

            # Mid-range with high volume spike = something happening
            elif 0.25 <= price <= 0.75 and is_very_active:
                reasoning.append(f"Uncertain price ${price:.2f} with HIGH activity")
                reasoning.append(f"24h volume spike: {volume_24h:,} contracts")
                reasoning.append("Market is actively debating - research opportunity")
                action = "HOLD"
                confidence = "MEDIUM"

            if reasoning and action != "HOLD":
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    title=title,
                    action=action,
                    confidence=confidence,
                    current_price=price,
                    target_price=None,
                    reasoning=reasoning,
                    category="volume_extreme",
                    data_sources=["kalshi"],
                    timestamp=now,
                    edge=edge,
                ))

        # Sort by volume
        recommendations.sort(key=lambda x: -(x.edge or 0))
        return recommendations[:10]

    def _analyze_expiring_markets(self) -> list[TradeRecommendation]:
        """Find markets expiring soon with uncertain prices."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            ticker = k_data.get("ticker", "UNKNOWN")
            title = k_data.get("title", "")
            price = k_data.get("yes_price")
            close_time = k_data.get("close_time") or k_data.get("expiration_time")

            if not price or not close_time:
                continue

            # Parse close time
            try:
                if isinstance(close_time, str):
                    # Handle ISO format
                    close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00').replace('+00:00', ''))
                else:
                    continue
            except:
                continue

            # Check if expiring within 7 days
            days_to_expiry = (close_dt - now).days
            if days_to_expiry < 0 or days_to_expiry > 7:
                continue

            # Look for uncertain prices on soon-expiring markets
            if 0.20 <= price <= 0.80:
                reasoning = [
                    f"Expires in {days_to_expiry} days with uncertain price ${price:.2f}",
                    "Market still undecided close to resolution",
                    "High potential for price movement before expiry",
                ]

                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    title=title,
                    action="HOLD",  # Research opportunity
                    confidence="MEDIUM",
                    current_price=price,
                    target_price=None,
                    reasoning=reasoning,
                    category="expiring_soon",
                    data_sources=["kalshi"],
                    timestamp=now,
                    edge=None,
                ))

        return recommendations[:5]

    def _analyze_news_momentum(self) -> list[TradeRecommendation]:
        """Find markets with relevant recent news."""
        recommendations = []
        now = datetime.utcnow()

        # Get recent news
        news = self.db.get_recent_news(hours=24, limit=200)
        if not news:
            return recommendations

        # Build keyword frequency
        keyword_news = {}
        important_keywords = [
            "trump", "biden", "election", "fed", "inflation", "recession",
            "tesla", "musk", "bitcoin", "crypto", "ai", "nvidia",
            "ukraine", "russia", "china", "war", "tariff",
        ]

        for item in news:
            title = (item.get("title") or "").lower()
            for kw in important_keywords:
                if kw in title:
                    if kw not in keyword_news:
                        keyword_news[kw] = []
                    keyword_news[kw].append(item.get("title", "")[:80])

        # Find markets related to trending topics
        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            ticker = k_data.get("ticker", "UNKNOWN")
            title = (k_data.get("title") or "").lower()
            price = k_data.get("yes_price")
            volume_24h = k_data.get("volume_24h") or 0

            if not price:
                continue

            # Check for keyword matches
            for kw, headlines in keyword_news.items():
                if len(headlines) >= 3 and kw in title:  # At least 3 news articles
                    reasoning = [
                        f"High news volume: {len(headlines)} articles mentioning '{kw}'",
                        f"Recent headlines:",
                    ]
                    for h in headlines[:3]:
                        reasoning.append(f"  • {h}")

                    if volume_24h > 1000:
                        reasoning.append(f"Active trading: {volume_24h:,} contracts in 24h")

                    recommendations.append(TradeRecommendation(
                        ticker=ticker,
                        title=k_data.get("title", ""),
                        action="HOLD",  # News-based, needs research
                        confidence="LOW",
                        current_price=price,
                        target_price=None,
                        reasoning=reasoning,
                        category="news_momentum",
                        data_sources=["kalshi", "rss"],
                        timestamp=now,
                        edge=None,
                    ))
                    break

        return recommendations[:5]

    def _find_betting_odds_arbitrage(self) -> list[TradeRecommendation]:
        """Find discrepancies between Kalshi and traditional betting markets."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        # Get betting odds data
        betting_data = self.db.get_latest_markets(source="betting_election", limit=100)
        betting_data.extend(self.db.get_latest_markets(source="betting_metaculus", limit=100))

        # Build betting odds index
        betting_index = {}
        for bd in betting_data:
            try:
                data = json.loads(bd.get("data", "{}"))
                candidate = (data.get("candidate") or data.get("title") or "").lower()
                probability = data.get("probability")
                if candidate and probability:
                    betting_index[candidate] = probability
            except:
                continue

        # Find matching Kalshi markets
        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            title = (k_data.get("title") or "").lower()
            ticker = k_data.get("ticker", "UNKNOWN")
            price = k_data.get("yes_price")

            if not price:
                continue

            # Check for candidate/event matches
            for candidate, bet_prob in betting_index.items():
                if candidate in title:
                    diff = abs(price - bet_prob)

                    if diff >= 0.05:  # 5%+ difference
                        edge = diff * 100

                        if price > bet_prob:
                            action = "BUY_NO"
                            reasoning = [
                                f"Kalshi YES @ ${price:.2f} vs betting markets @ ${bet_prob:.2f}",
                                f"Kalshi is {diff*100:.1f}% higher than betting consensus",
                                "Potential overpricing on Kalshi",
                            ]
                        else:
                            action = "BUY_YES"
                            reasoning = [
                                f"Kalshi YES @ ${price:.2f} vs betting markets @ ${bet_prob:.2f}",
                                f"Kalshi is {diff*100:.1f}% lower than betting consensus",
                                "Potential value on Kalshi",
                            ]

                        confidence = "MEDIUM" if diff >= 0.08 else "LOW"

                        recommendations.append(TradeRecommendation(
                            ticker=ticker,
                            title=k_data.get("title", ""),
                            action=action,
                            confidence=confidence,
                            current_price=price,
                            target_price=bet_prob,
                            reasoning=reasoning,
                            category="betting_arb",
                            data_sources=["kalshi", "betting_odds"],
                            timestamp=now,
                            edge=edge,
                        ))
                        break

        return recommendations[:10]

    def _analyze_calendar_events(self) -> list[TradeRecommendation]:
        """Find markets with upcoming catalyst events."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        # Get calendar events
        econ_calendar = self.db.get_latest_markets(source="calendar_fomc", limit=20)
        econ_calendar.extend(self.db.get_latest_markets(source="calendar_bls", limit=20))
        earnings_calendar = self.db.get_latest_markets(source="earnings_yahoo", limit=50)
        earnings_calendar.extend(self.db.get_latest_markets(source="earnings_nasdaq", limit=50))

        # Build event index by days until
        upcoming_fomc = []
        upcoming_bls = []
        upcoming_earnings = {}

        for event in econ_calendar:
            try:
                data = json.loads(event.get("data", "{}"))
                days_until = data.get("days_until")
                if days_until is not None and days_until <= 14:
                    event_type = data.get("type", "")
                    if "fomc" in event_type:
                        upcoming_fomc.append(data)
                    elif event_type in ("jobs_report", "cpi"):
                        upcoming_bls.append(data)
            except:
                continue

        for earning in earnings_calendar:
            try:
                data = json.loads(earning.get("data", "{}"))
                symbol = data.get("symbol")
                days_until = data.get("days_until")
                if symbol and days_until is not None and days_until <= 14:
                    upcoming_earnings[symbol.lower()] = data
            except:
                continue

        # Analyze markets for calendar catalysts
        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            title = (k_data.get("title") or "").lower()
            ticker = k_data.get("ticker", "UNKNOWN")
            price = k_data.get("yes_price")

            if not price:
                continue

            reasoning = []
            action = "HOLD"
            confidence = "LOW"
            edge = None

            # FOMC-related markets with upcoming meeting
            if upcoming_fomc and any(kw in title for kw in ["fed", "fomc", "rate", "powell"]):
                next_fomc = upcoming_fomc[0]
                days = next_fomc.get("days_until", 0)
                reasoning.append(f"FOMC meeting in {days} days")
                reasoning.append(f"Market at ${price:.2f} - expect volatility around decision")
                if 0.30 <= price <= 0.70:
                    reasoning.append("Uncertain price = high potential movement")
                    confidence = "MEDIUM"
                action = "HOLD"  # Flag for research

            # BLS releases affecting economic markets
            elif upcoming_bls and any(kw in title for kw in ["unemployment", "jobs", "cpi", "inflation", "payroll"]):
                next_bls = upcoming_bls[0]
                days = next_bls.get("days_until", 0)
                event_name = next_bls.get("name", "Economic release")
                reasoning.append(f"{event_name} in {days} days")
                reasoning.append(f"Market at ${price:.2f} - data release will move price")
                if 0.30 <= price <= 0.70:
                    reasoning.append("Uncertain price before data = opportunity")
                    confidence = "MEDIUM"
                action = "HOLD"

            # Earnings affecting company-related markets
            else:
                for company_kw in ["tesla", "apple", "nvidia", "microsoft", "google", "amazon", "meta"]:
                    symbol_map = {
                        "tesla": "tsla", "apple": "aapl", "nvidia": "nvda",
                        "microsoft": "msft", "google": "googl", "amazon": "amzn", "meta": "meta"
                    }
                    if company_kw in title:
                        symbol = symbol_map.get(company_kw)
                        if symbol and symbol in upcoming_earnings:
                            earnings = upcoming_earnings[symbol]
                            days = earnings.get("days_until", 0)
                            reasoning.append(f"{company_kw.title()} earnings in {days} days")
                            reasoning.append(f"Market at ${price:.2f} - earnings could be catalyst")
                            if 0.30 <= price <= 0.70:
                                reasoning.append("Position before earnings for volatility play")
                                confidence = "MEDIUM"
                            action = "HOLD"
                            break

            if reasoning:
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    title=k_data.get("title", ""),
                    action=action,
                    confidence=confidence,
                    current_price=price,
                    target_price=None,
                    reasoning=reasoning,
                    category="calendar_catalyst",
                    data_sources=["kalshi", "calendar"],
                    timestamp=now,
                    edge=edge,
                ))

        return recommendations[:10]

    def _analyze_social_sentiment(self) -> list[TradeRecommendation]:
        """Find markets with strong social sentiment signals."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = self.db.get_latest_markets(source="kalshi", limit=1000)

        # Get social sentiment data
        reddit_data = []
        for subreddit in ["wallstreetbets", "stocks", "investing", "politics", "technology"]:
            reddit_data.extend(self.db.get_latest_markets(source=f"reddit_{subreddit}", limit=50))

        hn_data = self.db.get_latest_markets(source="hackernews", limit=50)

        # Build sentiment keyword index
        keyword_sentiment = {}
        tracked_keywords = [
            "trump", "biden", "election", "fed", "inflation", "recession",
            "tesla", "musk", "bitcoin", "crypto", "ai", "nvidia",
            "rate", "unemployment", "jobs"
        ]

        for item in reddit_data + hn_data:
            try:
                data = json.loads(item.get("data", "{}"))
                keywords = data.get("keywords_found", [])
                sentiment = data.get("sentiment", "neutral")
                score = data.get("score", 0) or 0

                for kw in keywords:
                    if kw not in keyword_sentiment:
                        keyword_sentiment[kw] = {"positive": 0, "negative": 0, "neutral": 0, "total_score": 0}
                    keyword_sentiment[kw][sentiment] += 1
                    keyword_sentiment[kw]["total_score"] += score
            except:
                continue

        # Find markets with strong sentiment signal
        for km in kalshi_markets:
            k_data = json.loads(km.get("data", "{}"))
            title = (k_data.get("title") or "").lower()
            ticker = k_data.get("ticker", "UNKNOWN")
            price = k_data.get("yes_price")

            if not price:
                continue

            # Check for keyword matches with strong sentiment
            for kw, sentiment in keyword_sentiment.items():
                if kw in title:
                    total = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
                    if total < 5:
                        continue  # Not enough data

                    pos_ratio = sentiment["positive"] / total
                    neg_ratio = sentiment["negative"] / total

                    reasoning = []
                    action = "HOLD"
                    confidence = "LOW"
                    edge = None

                    # Strong bullish sentiment
                    if pos_ratio > 0.6 and sentiment["total_score"] > 1000:
                        reasoning.append(f"Strong bullish sentiment on '{kw}': {pos_ratio*100:.0f}% positive")
                        reasoning.append(f"Social score: {sentiment['total_score']:,} points across {total} posts")
                        if price < 0.50:
                            reasoning.append(f"Price ${price:.2f} may be undervalued given sentiment")
                            action = "BUY_YES"
                            confidence = "LOW"
                            edge = (0.55 - price) * 100

                    # Strong bearish sentiment
                    elif neg_ratio > 0.6 and sentiment["total_score"] > 1000:
                        reasoning.append(f"Strong bearish sentiment on '{kw}': {neg_ratio*100:.0f}% negative")
                        reasoning.append(f"Social score: {sentiment['total_score']:,} points across {total} posts")
                        if price > 0.50:
                            reasoning.append(f"Price ${price:.2f} may be overvalued given sentiment")
                            action = "BUY_NO"
                            confidence = "LOW"
                            edge = (price - 0.45) * 100

                    if reasoning:
                        recommendations.append(TradeRecommendation(
                            ticker=ticker,
                            title=k_data.get("title", ""),
                            action=action,
                            confidence=confidence,
                            current_price=price,
                            target_price=None,
                            reasoning=reasoning,
                            category="social_sentiment",
                            data_sources=["kalshi", "reddit", "hackernews"],
                            timestamp=now,
                            edge=edge,
                        ))
                        break  # One recommendation per market

        return recommendations[:10]

    def get_top_picks(self, n: int = 10) -> list[TradeRecommendation]:
        """Get top N actionable recommendations."""
        recs = self.generate_recommendations()
        # Filter to actionable (BUY_YES or BUY_NO)
        actionable = [r for r in recs if r.action in ("BUY_YES", "BUY_NO")]
        return actionable[:n]

    def get_research_opportunities(self, n: int = 10) -> list[TradeRecommendation]:
        """Get markets worth researching (HOLD recommendations)."""
        recs = self.generate_recommendations()
        research = [r for r in recs if r.action == "HOLD"]
        return research[:n]

    def print_strategy_report(self) -> str:
        """Generate a formatted strategy report."""
        recommendations = self.generate_recommendations()

        lines = [
            "=" * 70,
            "KALSHI INVESTMENT STRATEGY REPORT",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 70,
            "",
        ]

        # Get data summary
        kalshi_count = len(self.db.get_latest_markets(source="kalshi", limit=10000))
        poly_count = len(self.db.get_latest_markets(source="polymarket", limit=10000))
        indicator_count = len(self.db.get_latest_indicators())
        news_count = len(self.db.get_recent_news(hours=24))

        lines.append("DATA SOURCES")
        lines.append("-" * 40)
        lines.append(f"  Kalshi markets: {kalshi_count}")
        lines.append(f"  Polymarket markets: {poly_count}")
        lines.append(f"  Economic indicators: {indicator_count}")
        lines.append(f"  News articles (24h): {news_count}")
        lines.append("")

        # Summary by category
        categories = {}
        for rec in recommendations:
            cat = rec.category
            if cat not in categories:
                categories[cat] = {"BUY_YES": 0, "BUY_NO": 0, "HOLD": 0}
            categories[cat][rec.action] = categories[cat].get(rec.action, 0) + 1

        lines.append("OPPORTUNITIES BY CATEGORY")
        lines.append("-" * 40)
        for cat, actions in categories.items():
            total = sum(actions.values())
            lines.append(f"  {cat}: {total} ({actions.get('BUY_YES', 0)} YES, {actions.get('BUY_NO', 0)} NO, {actions.get('HOLD', 0)} research)")
        lines.append("")

        # Top picks
        top_picks = self.get_top_picks(10)
        if top_picks:
            lines.append("=" * 30 + " TOP PICKS " + "=" * 30)
            lines.append("")

            for i, rec in enumerate(top_picks, 1):
                conf = {"HIGH": "★★★", "MEDIUM": "★★☆", "LOW": "★☆☆"}.get(rec.confidence, "?")
                action = {"BUY_YES": "🟢 BUY YES", "BUY_NO": "🔴 BUY NO"}.get(rec.action, rec.action)
                edge_str = f" | Edge: {rec.edge:+.1f}%" if rec.edge else ""

                lines.append(f"{i}. {conf} {action} @ ${rec.current_price:.2f}{edge_str}")
                lines.append(f"   Ticker: {rec.ticker}")
                lines.append(f"   Market: {rec.title[:60]}...")
                lines.append(f"   Category: {rec.category}")
                for reason in rec.reasoning[:3]:
                    lines.append(f"   → {reason}")
                lines.append("")

        # Research opportunities
        research = self.get_research_opportunities(5)
        if research:
            lines.append("=" * 25 + " RESEARCH OPPORTUNITIES " + "=" * 25)
            lines.append("")

            for rec in research:
                lines.append(f"📊 {rec.ticker} @ ${rec.current_price:.2f}")
                lines.append(f"   {rec.title[:60]}...")
                for reason in rec.reasoning[:2]:
                    lines.append(f"   → {reason}")
                lines.append("")

        lines.append("=" * 70)
        lines.append("DISCLAIMER: Not financial advice. Do your own research.")
        lines.append("=" * 70)

        return "\n".join(lines)
