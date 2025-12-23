"""RSS feed aggregator for news sources."""

import logging
import hashlib
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class RSSCrawler(BaseCrawler):
    """Crawler for RSS/Atom news feeds."""

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="rss",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config
        self.feeds = config.rss_feeds
        self.seen_items: set[str] = set()  # Track seen item hashes

    def is_available(self) -> bool:
        """RSS feeds are publicly available."""
        return len(self.feeds) > 0

    def _parse_rss(self, xml_content: str) -> list[dict]:
        """Parse RSS 2.0 feed."""
        items = []
        try:
            root = ElementTree.fromstring(xml_content)

            # RSS 2.0 structure
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item"):
                    items.append({
                        "title": self._get_text(item, "title"),
                        "link": self._get_text(item, "link"),
                        "description": self._get_text(item, "description"),
                        "pubDate": self._get_text(item, "pubDate"),
                        "guid": self._get_text(item, "guid"),
                    })

        except ElementTree.ParseError as e:
            logger.error(f"RSS parse error: {e}")

        return items

    def _parse_atom(self, xml_content: str) -> list[dict]:
        """Parse Atom feed."""
        items = []
        try:
            root = ElementTree.fromstring(xml_content)

            # Atom namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                link = entry.find("atom:link", ns)
                link_href = link.get("href") if link is not None else None

                items.append({
                    "title": self._get_text_ns(entry, "atom:title", ns),
                    "link": link_href,
                    "description": self._get_text_ns(entry, "atom:summary", ns)
                    or self._get_text_ns(entry, "atom:content", ns),
                    "pubDate": self._get_text_ns(entry, "atom:published", ns)
                    or self._get_text_ns(entry, "atom:updated", ns),
                    "guid": self._get_text_ns(entry, "atom:id", ns),
                })

        except ElementTree.ParseError as e:
            logger.error(f"Atom parse error: {e}")

        return items

    def _get_text(self, element, tag: str) -> Optional[str]:
        """Get text content of a child element."""
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else None

    def _get_text_ns(self, element, tag: str, ns: dict) -> Optional[str]:
        """Get text content with namespace."""
        child = element.find(tag, ns)
        return child.text.strip() if child is not None and child.text else None

    def _item_hash(self, item: dict) -> str:
        """Generate a hash for deduplication."""
        content = f"{item.get('link', '')}{item.get('title', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def fetch_feed(self, feed_config: dict) -> list[dict]:
        """Fetch and parse a single RSS feed."""
        url = feed_config.get("url")
        if not url:
            return []

        try:
            response = self.get(url)
            content = response.text

            # Try RSS first, then Atom
            items = self._parse_rss(content)
            if not items:
                items = self._parse_atom(content)

            # Add metadata to each item
            for item in items:
                item["feed_name"] = feed_config.get("name", url)
                item["feed_category"] = feed_config.get("category", "general")
                item["feed_url"] = url

            return items

        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {url}: {e}")
            return []

    def crawl(self) -> list[CrawlResult]:
        """Crawl all configured RSS feeds."""
        results = []
        now = datetime.utcnow()

        for feed_config in self.feeds:
            try:
                items = self.fetch_feed(feed_config)

                for item in items:
                    # Check for duplicates
                    item_hash = self._item_hash(item)
                    is_new = item_hash not in self.seen_items

                    if is_new:
                        self.seen_items.add(item_hash)

                    result = CrawlResult(
                        source=f"rss_{feed_config.get('name', 'unknown')}",
                        category=feed_config.get("category", "general"),
                        data_type="news",
                        timestamp=now,
                        data={
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "description": self._clean_html(item.get("description", "")),
                            "published": item.get("pubDate"),
                            "feed_name": item.get("feed_name"),
                            "is_new": is_new,
                        },
                        metadata={
                            "guid": item.get("guid"),
                            "feed_url": item.get("feed_url"),
                            "item_hash": item_hash,
                        },
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"RSS crawl failed for {feed_config.get('name')}: {e}")
                self.error_count += 1

        self.last_crawl = now
        logger.info(f"RSS crawl complete: {len(results)} items from {len(self.feeds)} feeds")
        return results

    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags from content."""
        if not html_content:
            return ""

        import re
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", html_content)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        # Truncate
        return clean[:1000] if len(clean) > 1000 else clean

    def get_new_items(self) -> list[CrawlResult]:
        """Get only new items since last crawl."""
        all_results = self.crawl()
        return [r for r in all_results if r.data.get("is_new", False)]

    def add_feed(self, name: str, url: str, category: str = "general"):
        """Add a new RSS feed to monitor."""
        self.feeds.append({
            "name": name,
            "url": url,
            "category": category,
        })
        logger.info(f"Added RSS feed: {name} ({url})")

    def search_recent(self, keyword: str, limit: int = 50) -> list[dict]:
        """Search recent items for a keyword."""
        results = self.crawl()
        keyword_lower = keyword.lower()

        matching = []
        for result in results:
            title = result.data.get("title", "").lower()
            description = result.data.get("description", "").lower()

            if keyword_lower in title or keyword_lower in description:
                matching.append(result.data)

            if len(matching) >= limit:
                break

        return matching
