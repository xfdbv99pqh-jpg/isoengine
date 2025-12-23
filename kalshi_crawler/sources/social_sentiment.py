"""Social sentiment crawler for Reddit, HackerNews, and other platforms."""

import logging
import re
import json
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from ..base import BaseCrawler, CrawlResult
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class SocialSentimentCrawler(BaseCrawler):
    """Crawler for social media sentiment from Reddit, HackerNews, etc."""

    # Subreddits to monitor
    SUBREDDITS = [
        # Markets & Trading
        "wallstreetbets",
        "stocks",
        "investing",
        "options",
        # Economics
        "economics",
        "economy",
        # Politics
        "politics",
        "news",
        "worldnews",
        # Tech
        "technology",
        "nvidia",
        "teslamotors",
        # Crypto
        "cryptocurrency",
        "bitcoin",
    ]

    # Keywords to track
    TRACKED_KEYWORDS = [
        # Economic
        "fed", "fomc", "interest rate", "inflation", "cpi", "gdp", "recession",
        "unemployment", "jobs report", "payroll", "rate cut", "rate hike",
        # Political
        "election", "trump", "biden", "congress", "senate", "vote",
        # Tech/Corporate
        "tesla", "nvidia", "apple", "microsoft", "google", "meta",
        "ai", "chatgpt", "earnings",
        # Crypto
        "bitcoin", "btc", "ethereum", "eth", "crypto",
    ]

    def __init__(self, config: CrawlerConfig):
        super().__init__(
            name="social_sentiment",
            rate_limit=config.default_rate_limit,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            user_agent=config.user_agent,
        )
        self.config = config

    def is_available(self) -> bool:
        return True

    def _scrape_reddit_json(self, subreddit: str, limit: int = 25) -> list[dict]:
        """Scrape Reddit using .json endpoints (no auth required)."""
        results = []

        try:
            # Reddit's JSON API
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; KalshiCrawler/1.0; Educational Research)",
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    try:
                        post_data = post.get("data", {})

                        # Basic post info
                        result = {
                            "source": "reddit",
                            "subreddit": subreddit,
                            "id": post_data.get("id"),
                            "title": post_data.get("title"),
                            "selftext": post_data.get("selftext", "")[:500],  # Truncate
                            "url": f"https://reddit.com{post_data.get('permalink')}",
                            "score": post_data.get("score"),
                            "upvote_ratio": post_data.get("upvote_ratio"),
                            "num_comments": post_data.get("num_comments"),
                            "created_utc": post_data.get("created_utc"),
                            "scraped_at": datetime.utcnow().isoformat(),
                        }

                        # Analyze sentiment keywords
                        text = f"{result['title']} {result['selftext']}".lower()
                        result["keywords_found"] = [
                            kw for kw in self.TRACKED_KEYWORDS if kw in text
                        ]
                        result["has_tracked_keywords"] = len(result["keywords_found"]) > 0

                        # Simple sentiment analysis
                        result["sentiment"] = self._analyze_sentiment(text)

                        results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Reddit {subreddit} scrape failed: {e}")

        return results

    def _scrape_hackernews(self, limit: int = 30) -> list[dict]:
        """Scrape HackerNews front page."""
        results = []

        try:
            # HN API - get top stories
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = self.session.get(url, timeout=self.timeout)

            if response.status_code == 200:
                story_ids = response.json()[:limit]

                for story_id in story_ids:
                    try:
                        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        story_response = self.session.get(story_url, timeout=self.timeout)

                        if story_response.status_code == 200:
                            story = story_response.json()

                            result = {
                                "source": "hackernews",
                                "id": story.get("id"),
                                "title": story.get("title"),
                                "url": story.get("url"),
                                "hn_url": f"https://news.ycombinator.com/item?id={story.get('id')}",
                                "score": story.get("score"),
                                "num_comments": story.get("descendants", 0),
                                "by": story.get("by"),
                                "time": story.get("time"),
                                "scraped_at": datetime.utcnow().isoformat(),
                            }

                            # Check for tracked keywords
                            text = (result.get("title") or "").lower()
                            result["keywords_found"] = [
                                kw for kw in self.TRACKED_KEYWORDS if kw in text
                            ]
                            result["has_tracked_keywords"] = len(result["keywords_found"]) > 0

                            results.append(result)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"HackerNews scrape failed: {e}")

        return results

    def _scrape_twitter_trends(self) -> list[dict]:
        """Scrape Twitter/X trends - disabled as Nitter is mostly dead."""
        # Nitter instances are mostly down/blocked now
        # This is a placeholder for future X API integration
        return []

    def _analyze_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment analysis."""
        text = text.lower()

        positive_words = [
            "bullish", "moon", "pump", "buy", "long", "green", "up",
            "rally", "surge", "breakout", "good", "great", "amazing",
            "win", "winning", "gains", "profit", "optimistic"
        ]

        negative_words = [
            "bearish", "dump", "sell", "short", "red", "down", "crash",
            "plunge", "collapse", "bad", "terrible", "fear", "panic",
            "loss", "losing", "pessimistic", "recession", "bubble"
        ]

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count + 1:
            return "positive"
        elif neg_count > pos_count + 1:
            return "negative"
        else:
            return "neutral"

    def _get_sentiment_scores(self, posts: list[dict]) -> dict:
        """Aggregate sentiment from posts."""
        if not posts:
            return {"positive": 0, "neutral": 0, "negative": 0, "score": 0}

        sentiments = {"positive": 0, "neutral": 0, "negative": 0}

        for post in posts:
            sentiment = post.get("sentiment", "neutral")
            sentiments[sentiment] += 1

        total = sum(sentiments.values())
        if total == 0:
            return {"positive": 0, "neutral": 0, "negative": 0, "score": 0}

        # Calculate sentiment score (-1 to 1)
        score = (sentiments["positive"] - sentiments["negative"]) / total

        return {
            "positive": sentiments["positive"],
            "neutral": sentiments["neutral"],
            "negative": sentiments["negative"],
            "score": round(score, 2),
            "total_posts": total,
        }

    def crawl(self) -> list[CrawlResult]:
        """Crawl all social sentiment sources."""
        results = []
        now = datetime.utcnow()

        # Reddit - key subreddits
        for subreddit in self.SUBREDDITS[:8]:  # Limit to avoid rate limits
            try:
                posts = self._scrape_reddit_json(subreddit, limit=25)
                for item in posts:
                    if item.get("has_tracked_keywords"):  # Only save relevant posts
                        results.append(CrawlResult(
                            source=f"reddit_{subreddit}",
                            category=self._categorize_subreddit(subreddit),
                            data_type="social",
                            timestamp=now,
                            data=item,
                            metadata={
                                "subreddit": subreddit,
                                "sentiment": item.get("sentiment"),
                            },
                        ))
            except Exception as e:
                logger.error(f"Reddit {subreddit} crawl failed: {e}")
                self.error_count += 1

        # HackerNews
        try:
            posts = self._scrape_hackernews(limit=30)
            for item in posts:
                if item.get("has_tracked_keywords"):
                    results.append(CrawlResult(
                        source="hackernews",
                        category="tech",
                        data_type="social",
                        timestamp=now,
                        data=item,
                        metadata={"source": "hn"},
                    ))
        except Exception as e:
            logger.error(f"HackerNews crawl failed: {e}")
            self.error_count += 1

        # Twitter/Nitter (best effort)
        try:
            tweets = self._scrape_twitter_trends()
            for item in tweets:
                results.append(CrawlResult(
                    source="twitter",
                    category="social",
                    data_type="social",
                    timestamp=now,
                    data=item,
                    metadata={"search_term": item.get("search_term")},
                ))
        except Exception as e:
            logger.debug(f"Twitter crawl failed: {e}")
            self.error_count += 1

        self.last_crawl = now
        logger.info(f"Social sentiment crawl complete: {len(results)} results")
        return results

    def _categorize_subreddit(self, subreddit: str) -> str:
        """Categorize subreddit."""
        markets = ["wallstreetbets", "stocks", "investing", "options"]
        economics = ["economics", "economy"]
        politics = ["politics", "news", "worldnews"]
        tech = ["technology", "nvidia", "teslamotors"]
        crypto = ["cryptocurrency", "bitcoin"]

        if subreddit in markets:
            return "markets"
        elif subreddit in economics:
            return "economics"
        elif subreddit in politics:
            return "politics"
        elif subreddit in tech:
            return "tech"
        elif subreddit in crypto:
            return "crypto"
        return "other"

    def get_sentiment_summary(self) -> dict:
        """Get aggregated sentiment summary."""
        results = self.crawl()

        # Group by category
        by_category = {}
        for result in results:
            cat = result.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result.data)

        # Calculate sentiment per category
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "by_category": {},
            "overall": {},
        }

        all_posts = []
        for cat, posts in by_category.items():
            summary["by_category"][cat] = self._get_sentiment_scores(posts)
            all_posts.extend(posts)

        summary["overall"] = self._get_sentiment_scores(all_posts)

        return summary

    def get_trending_topics(self) -> list[dict]:
        """Get trending topics by keyword frequency."""
        results = self.crawl()

        # Count keyword occurrences
        keyword_counts = {}
        for result in results:
            for kw in result.data.get("keywords_found", []):
                if kw not in keyword_counts:
                    keyword_counts[kw] = 0
                keyword_counts[kw] += 1

        # Sort by count
        trending = sorted(
            [{"keyword": kw, "count": count} for kw, count in keyword_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        return trending[:20]
