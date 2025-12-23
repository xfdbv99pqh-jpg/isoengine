"""Base crawler class with rate limiting and retry logic."""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from collections import deque

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Standard result format for all crawlers."""

    source: str
    category: str
    data_type: str  # "market", "indicator", "news", "poll", etc.
    timestamp: datetime
    data: dict[str, Any]
    raw_response: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "source": self.source,
            "category": self.category,
            "data_type": self.data_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.request_times: deque = deque(maxlen=requests_per_minute)

    def wait_if_needed(self):
        """Block until we're allowed to make another request."""
        now = time.time()

        # Clean old requests
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        # Also ensure minimum interval between requests
        if self.request_times:
            elapsed = now - self.request_times[-1]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

        self.request_times.append(time.time())


class BaseCrawler(ABC):
    """Base class for all data source crawlers."""

    def __init__(
        self,
        name: str,
        rate_limit: int = 30,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "KalshiCrawler/0.1",
    ):
        self.name = name
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.session = self._create_session()
        self.last_crawl: Optional[datetime] = None
        self.crawl_count = 0
        self.error_count = 0

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})

        return session

    def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """Make a rate-limited request with error handling."""
        self.rate_limiter.wait_if_needed()

        kwargs.setdefault("timeout", self.timeout)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            self.crawl_count += 1
            return response
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"[{self.name}] Request failed: {e}")
            raise

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self._request("POST", url, **kwargs)

    @abstractmethod
    def crawl(self) -> list[CrawlResult]:
        """
        Execute the crawl and return results.

        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is available (has required config, etc.)."""
        pass

    def get_status(self) -> dict:
        """Get crawler status for monitoring."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "last_crawl": self.last_crawl.isoformat() if self.last_crawl else None,
            "crawl_count": self.crawl_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.crawl_count),
        }
