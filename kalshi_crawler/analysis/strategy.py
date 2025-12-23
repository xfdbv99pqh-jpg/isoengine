"""
Kalshi Investment Strategy Generator

Analyzes crawled data to generate actionable trading recommendations.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass
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

    def __str__(self):
        arrow = "↑" if self.action == "BUY_YES" else "↓" if self.action == "BUY_NO" else "→"
        price_str = f"${self.current_price:.2f}" if self.current_price else "N/A"
        return f"[{self.confidence}] {arrow} {self.action} {self.ticker} @ {price_str}\n    {self.title[:60]}..."

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
        }


class StrategyGenerator:
    """Generates Kalshi trading strategies based on collected data."""

    # Indicator thresholds for economic markets
    INDICATOR_THRESHOLDS = {
        "UNRATE": {"bullish_below": 4.0, "bearish_above": 5.0},  # Unemployment
        "CPIAUCSL": {"yoy_bullish_below": 2.5, "yoy_bearish_above": 3.5},  # CPI
        "FEDFUNDS": {"cut_signal": -0.25, "hike_signal": 0.25},  # Fed Funds
        "DGS10": {"low": 4.0, "high": 5.0},  # 10Y Treasury
        "T10Y2Y": {"inversion_warning": 0, "normal": 0.5},  # Yield curve
    }

    def __init__(self, db: CrawlerDatabase):
        self.db = db

    def generate_recommendations(self) -> list[TradeRecommendation]:
        """Generate all trading recommendations."""
        recommendations = []

        # Economic-based recommendations
        recommendations.extend(self._analyze_economic_markets())

        # Cross-market arbitrage opportunities
        recommendations.extend(self._analyze_arbitrage())

        # News-driven opportunities
        recommendations.extend(self._analyze_news_catalysts())

        # Sort by confidence
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: confidence_order.get(x.confidence, 3))

        return recommendations

    def _analyze_economic_markets(self) -> list[TradeRecommendation]:
        """Analyze economic indicators vs market prices."""
        recommendations = []
        now = datetime.utcnow()

        # Get latest indicators
        indicators = {}
        for ind in self.db.get_latest_indicators():
            indicators[ind["series_id"]] = ind

        # Get Polymarket economics markets (since Kalshi might not be available)
        markets = self.db.get_latest_markets(category="economics", limit=100)

        # Also check all markets for economic keywords
        all_markets = self.db.get_latest_markets(limit=500)

        for market in all_markets:
            data = json.loads(market.get("data", "{}"))
            title = (data.get("title") or data.get("question") or "").lower()
            ticker = market.get("ticker") or data.get("condition_id") or "UNKNOWN"

            price = data.get("yes_price")
            if not price:
                prices = data.get("prices", {})
                price = prices.get("Yes") or prices.get("yes")

            reasoning = []
            action = "HOLD"
            confidence = "LOW"
            sources = [market.get("source", "unknown")]

            # CPI / Inflation markets
            if any(kw in title for kw in ["inflation", "cpi", "consumer price"]):
                cpi = indicators.get("CPIAUCSL")
                core_cpi = indicators.get("CPILFESL")

                if cpi:
                    cpi_data = json.loads(cpi.get("data", "{}"))
                    yoy_change = cpi_data.get("change_12_period", {}).get("percent")

                    if yoy_change is not None:
                        sources.append("FRED:CPIAUCSL")
                        if yoy_change > 3.5:
                            reasoning.append(f"CPI YoY at {yoy_change:.1f}% - elevated inflation")
                            if "above" in title or "higher" in title:
                                action = "BUY_YES"
                                confidence = "MEDIUM"
                            elif "below" in title or "lower" in title:
                                action = "BUY_NO"
                                confidence = "MEDIUM"
                        elif yoy_change < 2.5:
                            reasoning.append(f"CPI YoY at {yoy_change:.1f}% - inflation cooling")
                            if "below" in title or "lower" in title:
                                action = "BUY_YES"
                                confidence = "MEDIUM"

            # Fed Rate markets - be specific to avoid matching "federal spending" etc.
            elif any(kw in title for kw in ["fomc", "interest rate", "rate cut", "rate hike", "federal reserve", "fed funds", "fed rate"]) or ("fed" in title.split() and "federal spending" not in title):
                fed_funds = indicators.get("FEDFUNDS")
                t10y2y = indicators.get("T10Y2Y")

                if fed_funds:
                    ff_data = json.loads(fed_funds.get("data", "{}"))
                    ff_change = ff_data.get("change_1_period", {}).get("absolute")
                    current_rate = ff_data.get("current_value")

                    sources.append("FRED:FEDFUNDS")
                    if current_rate:
                        reasoning.append(f"Fed Funds currently at {current_rate:.2f}%")

                    if ff_change and ff_change < -0.2:
                        reasoning.append("Recent rate cut detected")
                        if "cut" in title:
                            action = "BUY_YES"
                            confidence = "MEDIUM"

                if t10y2y:
                    spread_data = json.loads(t10y2y.get("data", "{}"))
                    spread = spread_data.get("current_value")
                    if spread is not None and spread < 0:
                        reasoning.append(f"Yield curve inverted ({spread:.2f}%) - recession signal")
                        sources.append("FRED:T10Y2Y")

            # Unemployment markets
            elif any(kw in title for kw in ["unemployment", "jobless", "jobs report", "payroll"]):
                unrate = indicators.get("UNRATE")
                claims = indicators.get("ICSA")

                if unrate:
                    ur_data = json.loads(unrate.get("data", "{}"))
                    current_ur = ur_data.get("current_value")
                    ur_change = ur_data.get("change_1_period", {}).get("absolute")

                    if current_ur:
                        sources.append("FRED:UNRATE")
                        reasoning.append(f"Unemployment at {current_ur:.1f}%")

                        if current_ur < 4.0:
                            reasoning.append("Labor market tight")
                            if "below" in title or "under" in title:
                                action = "BUY_YES"
                                confidence = "MEDIUM"
                        elif current_ur > 5.0:
                            reasoning.append("Labor market weakening")
                            if "above" in title or "over" in title:
                                action = "BUY_YES"
                                confidence = "MEDIUM"

            # GDP markets
            elif any(kw in title for kw in ["gdp", "recession", "economic growth"]):
                gdp = indicators.get("GDPC1")

                if gdp:
                    gdp_data = json.loads(gdp.get("data", "{}"))
                    gdp_change = gdp_data.get("change_1_period", {}).get("percent")

                    if gdp_change is not None:
                        sources.append("FRED:GDPC1")
                        if gdp_change < 0:
                            reasoning.append(f"GDP contracted {gdp_change:.1f}%")
                            if "recession" in title:
                                action = "BUY_YES"
                                confidence = "MEDIUM"
                        elif gdp_change > 2:
                            reasoning.append(f"GDP growing at {gdp_change:.1f}%")
                            if "recession" in title:
                                action = "BUY_NO"
                                confidence = "LOW"

            if reasoning and action != "HOLD":
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    title=data.get("title") or data.get("question", ""),
                    action=action,
                    confidence=confidence,
                    current_price=price,
                    target_price=None,
                    reasoning=reasoning,
                    category="economics",
                    data_sources=sources,
                    timestamp=now,
                ))

        return recommendations

    def _analyze_arbitrage(self) -> list[TradeRecommendation]:
        """Find arbitrage between Kalshi and Polymarket."""
        recommendations = []
        now = datetime.utcnow()

        kalshi_markets = {
            m["ticker"]: m for m in self.db.get_latest_markets(source="kalshi")
            if m.get("ticker")
        }
        poly_markets = self.db.get_latest_markets(source="polymarket", limit=300)

        # Look for similar markets with price discrepancies
        for poly in poly_markets:
            p_data = json.loads(poly.get("data", "{}"))
            p_question = (p_data.get("question") or "").lower()
            p_prices = p_data.get("prices", {})
            p_price = p_prices.get("Yes") or p_prices.get("yes")

            if not p_price or not p_question:
                continue

            # Search for matching Kalshi markets
            for k_ticker, kalshi in kalshi_markets.items():
                k_data = json.loads(kalshi.get("data", "{}"))
                k_title = (k_data.get("title") or "").lower()
                k_price = k_data.get("yes_price")

                if not k_price:
                    continue

                # Check for similar topics
                common_words = set(p_question.split()) & set(k_title.split())
                significant_words = [w for w in common_words if len(w) > 4]

                if len(significant_words) >= 3:  # At least 3 significant matching words
                    price_diff = abs(k_price - p_price)

                    if price_diff >= 0.05:  # 5% or more difference
                        higher_platform = "Kalshi" if k_price > p_price else "Polymarket"
                        lower_platform = "Polymarket" if k_price > p_price else "Kalshi"

                        recommendations.append(TradeRecommendation(
                            ticker=k_ticker,
                            title=k_data.get("title", ""),
                            action="BUY_YES" if k_price < p_price else "BUY_NO",
                            confidence="HIGH" if price_diff >= 0.10 else "MEDIUM",
                            current_price=k_price,
                            target_price=p_price,
                            reasoning=[
                                f"Price discrepancy: {higher_platform} @ ${max(k_price, p_price):.2f} vs {lower_platform} @ ${min(k_price, p_price):.2f}",
                                f"Potential {price_diff*100:.1f}% arbitrage opportunity",
                                f"Similar market on Polymarket: {p_data.get('question', '')[:50]}...",
                            ],
                            category="arbitrage",
                            data_sources=["kalshi", "polymarket"],
                            timestamp=now,
                        ))

        return recommendations

    def _analyze_news_catalysts(self) -> list[TradeRecommendation]:
        """Find markets affected by recent news."""
        recommendations = []
        now = datetime.utcnow()

        # Get recent news with important keywords
        news = self.db.get_recent_news(hours=12, limit=200)
        markets = self.db.get_latest_markets(limit=500)

        # Keywords that suggest market-moving news
        catalyst_keywords = {
            "fed": ["fed", "fomc", "powell", "interest rate"],
            "inflation": ["inflation", "cpi", "prices", "consumer"],
            "jobs": ["jobs", "employment", "unemployment", "payroll", "labor"],
            "election": ["election", "vote", "poll", "candidate", "trump", "biden"],
            "tech": ["tesla", "musk", "apple", "google", "ai", "nvidia"],
        }

        # Count keyword mentions in recent news
        keyword_counts = {cat: 0 for cat in catalyst_keywords}
        keyword_headlines = {cat: [] for cat in catalyst_keywords}

        for item in news:
            title = (item.get("title") or "").lower()
            content = (item.get("content") or "").lower()
            text = f"{title} {content}"

            for category, keywords in catalyst_keywords.items():
                if any(kw in text for kw in keywords):
                    keyword_counts[category] += 1
                    if title and len(keyword_headlines[category]) < 3:
                        keyword_headlines[category].append(item.get("title", "")[:80])

        # Find markets related to trending news topics
        for market in markets:
            data = json.loads(market.get("data", "{}"))
            title = (data.get("title") or data.get("question") or "").lower()
            ticker = market.get("ticker") or data.get("condition_id") or "UNKNOWN"

            price = data.get("yes_price")
            if not price:
                prices = data.get("prices", {})
                price = prices.get("Yes") or prices.get("yes")

            for category, keywords in catalyst_keywords.items():
                if keyword_counts[category] >= 3:  # Significant news volume
                    if any(kw in title for kw in keywords):
                        recommendations.append(TradeRecommendation(
                            ticker=ticker,
                            title=data.get("title") or data.get("question", ""),
                            action="HOLD",  # News-based, needs manual review
                            confidence="LOW",
                            current_price=price,
                            target_price=None,
                            reasoning=[
                                f"High news volume ({keyword_counts[category]} articles) for '{category}'",
                                "Recent headlines:",
                                *[f"  • {h}" for h in keyword_headlines[category]],
                                "Review news before trading - catalyst detected",
                            ],
                            category="news_catalyst",
                            data_sources=["rss_feeds", market.get("source", "unknown")],
                            timestamp=now,
                        ))
                        break  # One recommendation per market

        return recommendations

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

        # Summary
        action_counts = {}
        for rec in recommendations:
            action_counts[rec.action] = action_counts.get(rec.action, 0) + 1

        lines.append("SUMMARY")
        lines.append("-" * 40)
        for action, count in sorted(action_counts.items()):
            lines.append(f"  {action}: {count}")
        lines.append("")

        # Group by category
        categories = {}
        for rec in recommendations:
            if rec.category not in categories:
                categories[rec.category] = []
            categories[rec.category].append(rec)

        for category, recs in categories.items():
            lines.append(f"\n{'='*30} {category.upper()} {'='*30}")
            lines.append("")

            for rec in recs[:10]:  # Top 10 per category
                confidence_symbol = {"HIGH": "★★★", "MEDIUM": "★★☆", "LOW": "★☆☆"}.get(rec.confidence, "☆☆☆")
                action_symbol = {"BUY_YES": "🟢 BUY YES", "BUY_NO": "🔴 BUY NO", "HOLD": "🟡 WATCH", "AVOID": "⚫ AVOID"}.get(rec.action, rec.action)

                lines.append(f"{confidence_symbol} {action_symbol}")
                lines.append(f"   Ticker: {rec.ticker}")
                lines.append(f"   Market: {rec.title[:65]}...")
                if rec.current_price:
                    lines.append(f"   Price:  ${rec.current_price:.2f}")
                if rec.target_price:
                    lines.append(f"   Target: ${rec.target_price:.2f}")
                lines.append("   Reasoning:")
                for reason in rec.reasoning:
                    lines.append(f"      • {reason}")
                lines.append(f"   Sources: {', '.join(rec.data_sources)}")
                lines.append("")

        lines.append("=" * 70)
        lines.append("DISCLAIMER: This is not financial advice. Do your own research.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def get_top_picks(self, n: int = 5) -> list[TradeRecommendation]:
        """Get top N recommendations by confidence."""
        recs = self.generate_recommendations()
        # Filter to actionable (not HOLD)
        actionable = [r for r in recs if r.action in ("BUY_YES", "BUY_NO")]
        return actionable[:n]
