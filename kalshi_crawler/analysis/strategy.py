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

        # 2. Economic indicator misalignment
        recommendations.extend(self._analyze_economic_misalignment())

        # 3. High volume + price extreme opportunities
        recommendations.extend(self._analyze_volume_price_extremes())

        # 4. Expiring soon with uncertain prices
        recommendations.extend(self._analyze_expiring_markets())

        # 5. News-driven momentum
        recommendations.extend(self._analyze_news_momentum())

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
            cpi_yoy = cpi_data.get("change_12_period", {}).get("percent")

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
        """Find liquid markets with extreme prices."""
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

            # Require significant liquidity
            is_liquid = volume > 50000 or (volume > 10000 and volume_24h > 2000)
            is_very_liquid = volume > 100000 or volume_24h > 10000

            if not is_liquid:
                continue

            reasoning = []
            action = "HOLD"
            confidence = "LOW"
            edge = None

            # Very low price with high volume = market confident it won't happen
            if price <= 0.10 and is_very_liquid:
                reasoning.append(f"Price ${price:.2f} - market says only {price*100:.0f}% chance")
                reasoning.append(f"Very liquid: {volume:,} total / {volume_24h:,} 24h volume")
                reasoning.append("Contrarian YES if you have edge the market is missing")
                action = "BUY_YES"
                confidence = "LOW"
                edge = (0.20 - price) * 100  # If true probability is 20%

            # Very high price with high volume
            elif price >= 0.90 and is_very_liquid:
                reasoning.append(f"Price ${price:.2f} - market says {price*100:.0f}% likely")
                reasoning.append(f"Very liquid: {volume:,} total / {volume_24h:,} 24h volume")
                reasoning.append("Contrarian NO if you see risk the market is missing")
                action = "BUY_NO"
                confidence = "LOW"
                edge = (price - 0.80) * 100  # If true probability is 80%

            # Mid-range with unusual volume spike
            elif 0.30 <= price <= 0.70 and volume_24h > 5000:
                reasoning.append(f"Uncertain price ${price:.2f} with high activity")
                reasoning.append(f"24h volume spike: {volume_24h:,} contracts")
                reasoning.append("Something may be happening - research this market")
                action = "HOLD"  # Flag for research, don't recommend action
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
