"""
Kalshi Web Crawler - A modular data aggregation system for informed trading decisions.

This package provides:
- Multi-source data collection (APIs, RSS, web scraping)
- Rate-limited, respectful crawling
- Signal extraction and cross-market comparison
- Local storage with SQLite
"""

from .config import CrawlerConfig
from .base import BaseCrawler, CrawlResult
from .runner import CrawlerRunner

__version__ = "0.1.0"
__all__ = ["CrawlerConfig", "BaseCrawler", "CrawlResult", "CrawlerRunner"]
