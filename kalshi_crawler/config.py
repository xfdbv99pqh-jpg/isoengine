"""Configuration management for the Kalshi crawler system."""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class CrawlerConfig:
    """Central configuration for all crawlers."""

    # API Keys (environment variables override defaults)
    kalshi_api_key: Optional[str] = field(default_factory=lambda: os.getenv("KALSHI_API_KEY"))
    kalshi_api_secret: Optional[str] = field(default_factory=lambda: os.getenv("KALSHI_API_SECRET"))
    fred_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FRED_API_KEY", "98c03a5ce94055b8d476157e6363f35c"))

    # Storage
    db_path: Path = field(default_factory=lambda: Path("./kalshi_data.db"))

    # Rate limiting (requests per minute)
    default_rate_limit: int = 30
    kalshi_rate_limit: int = 10
    fred_rate_limit: int = 120
    polymarket_rate_limit: int = 30

    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Request settings
    request_timeout: int = 30
    user_agent: str = "KalshiCrawler/0.1 (Educational Research)"

    # Kalshi API settings
    kalshi_base_url: str = "https://trading-api.kalshi.com/trade-api/v2"
    kalshi_demo_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    use_kalshi_demo: bool = True  # Start with demo API

    # FRED settings
    fred_base_url: str = "https://api.stlouisfed.org/fred"

    # Polymarket settings
    polymarket_base_url: str = "https://gamma-api.polymarket.com"

    # RSS feeds to monitor
    rss_feeds: list = field(default_factory=lambda: [
        # Politics
        {"name": "AP Politics", "url": "https://rsshub.app/apnews/topics/politics", "category": "politics"},
        {"name": "Reuters Politics", "url": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best&best-topics=political-general", "category": "politics"},
        # Economics
        {"name": "Fed News", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "category": "economics"},
        # Tech
        {"name": "HackerNews", "url": "https://hnrss.org/frontpage", "category": "tech"},
        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech"},
    ])

    # Polling sources
    polling_sources: list = field(default_factory=lambda: [
        "538",
        "rcp",  # RealClearPolitics
    ])

    # Key FRED series to track
    fred_series: list = field(default_factory=lambda: [
        # Employment
        {"id": "UNRATE", "name": "Unemployment Rate", "category": "employment"},
        {"id": "PAYEMS", "name": "Nonfarm Payrolls", "category": "employment"},
        {"id": "ICSA", "name": "Initial Claims", "category": "employment"},
        # Inflation
        {"id": "CPIAUCSL", "name": "CPI All Items", "category": "inflation"},
        {"id": "CPILFESL", "name": "Core CPI", "category": "inflation"},
        {"id": "PCEPI", "name": "PCE Price Index", "category": "inflation"},
        # GDP & Growth
        {"id": "GDP", "name": "GDP", "category": "growth"},
        {"id": "GDPC1", "name": "Real GDP", "category": "growth"},
        # Interest Rates
        {"id": "FEDFUNDS", "name": "Fed Funds Rate", "category": "rates"},
        {"id": "DGS10", "name": "10-Year Treasury", "category": "rates"},
        {"id": "DGS2", "name": "2-Year Treasury", "category": "rates"},
        {"id": "T10Y2Y", "name": "10Y-2Y Spread", "category": "rates"},
    ])

    @classmethod
    def from_env(cls) -> "CrawlerConfig":
        """Create config from environment variables."""
        return cls(
            kalshi_api_key=os.getenv("KALSHI_API_KEY"),
            kalshi_api_secret=os.getenv("KALSHI_API_SECRET"),
            fred_api_key=os.getenv("FRED_API_KEY", "98c03a5ce94055b8d476157e6363f35c"),
            db_path=Path(os.getenv("CRAWLER_DB_PATH", "./kalshi_data.db")),
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of warnings."""
        warnings = []

        if not self.kalshi_api_key:
            warnings.append("KALSHI_API_KEY not set - Kalshi source will be limited")
        if not self.fred_api_key:
            warnings.append("FRED_API_KEY not set - FRED source will be disabled")

        return warnings
