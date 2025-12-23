"""Data source crawlers."""

from .kalshi import KalshiCrawler
from .fred import FredCrawler
from .polymarket import PolymarketCrawler
from .polling import PollingCrawler
from .rss import RSSCrawler
from .economic_calendar import EconomicCalendarCrawler
from .betting_odds import BettingOddsCrawler
from .earnings_calendar import EarningsCalendarCrawler
from .social_sentiment import SocialSentimentCrawler

__all__ = [
    "KalshiCrawler",
    "FredCrawler",
    "PolymarketCrawler",
    "PollingCrawler",
    "RSSCrawler",
    "EconomicCalendarCrawler",
    "BettingOddsCrawler",
    "EarningsCalendarCrawler",
    "SocialSentimentCrawler",
]
