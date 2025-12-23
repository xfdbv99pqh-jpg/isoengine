"""SQLite storage for crawler data with time-series capabilities."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

from ..base import CrawlResult

logger = logging.getLogger(__name__)


class CrawlerDatabase:
    """SQLite database for storing crawler results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Market data (Kalshi, Polymarket)
                CREATE TABLE IF NOT EXISTS markets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    ticker TEXT,
                    title TEXT,
                    category TEXT,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data JSON NOT NULL,
                    metadata JSON,
                    UNIQUE(source, ticker, crawled_at)
                );

                CREATE INDEX IF NOT EXISTS idx_markets_source ON markets(source);
                CREATE INDEX IF NOT EXISTS idx_markets_ticker ON markets(ticker);
                CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
                CREATE INDEX IF NOT EXISTS idx_markets_crawled_at ON markets(crawled_at);

                -- Price history for tracking changes
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    price REAL,
                    volume INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_history(ticker);
                CREATE INDEX IF NOT EXISTS idx_price_recorded ON price_history(recorded_at);

                -- Economic indicators (FRED)
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    name TEXT,
                    category TEXT,
                    value REAL,
                    value_date TEXT,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data JSON,
                    UNIQUE(source, series_id, value_date)
                );

                CREATE INDEX IF NOT EXISTS idx_indicators_series ON indicators(series_id);
                CREATE INDEX IF NOT EXISTS idx_indicators_category ON indicators(category);

                -- News items (RSS, scraped)
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT,
                    link TEXT,
                    category TEXT,
                    published_at TIMESTAMP,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content TEXT,
                    item_hash TEXT UNIQUE,
                    metadata JSON
                );

                CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);
                CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);
                CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);

                -- Polling data
                CREATE TABLE IF NOT EXISTS polls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    poll_type TEXT,
                    data JSON NOT NULL,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_polls_source ON polls(source);
                CREATE INDEX IF NOT EXISTS idx_polls_type ON polls(poll_type);

                -- Signals and alerts
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,
                    source TEXT,
                    ticker TEXT,
                    message TEXT,
                    severity TEXT DEFAULT 'info',
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
                CREATE INDEX IF NOT EXISTS idx_signals_severity ON signals(severity);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

                -- Crawl log for monitoring
                CREATE TABLE IF NOT EXISTS crawl_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    items_count INTEGER,
                    error_count INTEGER,
                    status TEXT,
                    message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_crawl_source ON crawl_log(source);
                CREATE INDEX IF NOT EXISTS idx_crawl_started ON crawl_log(started_at);
            """)
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_result(self, result: CrawlResult):
        """Store a single crawl result in the appropriate table."""
        with self._get_connection() as conn:
            if result.data_type == "market":
                self._store_market(conn, result)
            elif result.data_type == "indicator":
                self._store_indicator(conn, result)
            elif result.data_type == "news":
                self._store_news(conn, result)
            elif result.data_type == "poll":
                self._store_poll(conn, result)
            elif result.data_type in ("calendar", "betting_odds", "social"):
                # Store calendar, betting, and social data in markets table
                self._store_generic(conn, result)
            else:
                logger.warning(f"Unknown data type: {result.data_type}")

    def store_results(self, results: list[CrawlResult]):
        """Store multiple crawl results."""
        for result in results:
            try:
                self.store_result(result)
            except Exception as e:
                logger.error(f"Failed to store result: {e}")

    def _store_market(self, conn: sqlite3.Connection, result: CrawlResult):
        """Store market data."""
        data = result.data
        ticker = data.get("ticker") or data.get("condition_id")

        conn.execute("""
            INSERT OR REPLACE INTO markets (source, ticker, title, category, crawled_at, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.source,
            ticker,
            data.get("title") or data.get("question"),
            result.category,
            result.timestamp,
            json.dumps(data),
            json.dumps(result.metadata),
        ))

        # Store price history
        price = data.get("yes_price") or (data.get("prices", {}).get("Yes"))
        if price is not None:
            conn.execute("""
                INSERT INTO price_history (source, ticker, price, volume, recorded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                result.source,
                ticker,
                price,
                data.get("volume"),
                result.timestamp,
            ))

    def _store_indicator(self, conn: sqlite3.Connection, result: CrawlResult):
        """Store economic indicator data."""
        data = result.data
        conn.execute("""
            INSERT OR REPLACE INTO indicators
            (source, series_id, name, category, value, value_date, crawled_at, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.source,
            data.get("series_id"),
            data.get("name"),
            result.category,
            data.get("current_value"),
            data.get("current_date"),
            result.timestamp,
            json.dumps(data),
        ))

    def _store_news(self, conn: sqlite3.Connection, result: CrawlResult):
        """Store news item."""
        data = result.data
        conn.execute("""
            INSERT OR IGNORE INTO news
            (source, title, link, category, published_at, crawled_at, content, item_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.source,
            data.get("title"),
            data.get("link"),
            result.category,
            data.get("published"),
            result.timestamp,
            data.get("description"),
            result.metadata.get("item_hash"),
            json.dumps(result.metadata),
        ))

    def _store_poll(self, conn: sqlite3.Connection, result: CrawlResult):
        """Store polling data."""
        conn.execute("""
            INSERT INTO polls (source, poll_type, data, crawled_at)
            VALUES (?, ?, ?, ?)
        """, (
            result.source,
            result.data.get("poll_type"),
            json.dumps(result.data),
            result.timestamp,
        ))

    def _store_generic(self, conn: sqlite3.Connection, result: CrawlResult):
        """Store generic data (calendar, betting_odds, social) in markets table."""
        data = result.data
        # Use source as identifier since these don't have tickers
        ticker = data.get("id") or data.get("symbol") or data.get("event") or result.source

        conn.execute("""
            INSERT OR REPLACE INTO markets (source, ticker, title, category, crawled_at, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.source,
            str(ticker)[:100],  # Limit length
            data.get("title") or data.get("name") or data.get("event") or data.get("question"),
            result.category,
            result.timestamp,
            json.dumps(data),
            json.dumps(result.metadata),
        ))

    def store_signal(
        self,
        signal_type: str,
        message: str,
        source: Optional[str] = None,
        ticker: Optional[str] = None,
        severity: str = "info",
        data: Optional[dict] = None,
    ):
        """Store a signal or alert."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO signals (signal_type, source, ticker, message, severity, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (signal_type, source, ticker, message, severity, json.dumps(data or {})))

    def log_crawl(
        self,
        source: str,
        started_at: datetime,
        completed_at: datetime,
        items_count: int,
        error_count: int,
        status: str,
        message: Optional[str] = None,
    ):
        """Log a crawl execution."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO crawl_log (source, started_at, completed_at, items_count, error_count, status, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source, started_at, completed_at, items_count, error_count, status, message))

    # Query methods
    def get_latest_markets(self, source: Optional[str] = None, category: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Get the latest market data."""
        with self._get_connection() as conn:
            query = "SELECT * FROM markets WHERE 1=1"
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)
            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY crawled_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_price_history(self, ticker: str, hours: int = 24) -> list[dict]:
        """Get price history for a ticker."""
        with self._get_connection() as conn:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            rows = conn.execute("""
                SELECT * FROM price_history
                WHERE ticker = ? AND recorded_at > ?
                ORDER BY recorded_at ASC
            """, (ticker, cutoff)).fetchall()
            return [dict(row) for row in rows]

    def get_latest_indicators(self, category: Optional[str] = None) -> list[dict]:
        """Get the latest economic indicators."""
        with self._get_connection() as conn:
            if category:
                rows = conn.execute("""
                    SELECT * FROM indicators
                    WHERE category = ?
                    GROUP BY series_id
                    HAVING crawled_at = MAX(crawled_at)
                """, (category,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM indicators
                    GROUP BY series_id
                    HAVING crawled_at = MAX(crawled_at)
                """).fetchall()
            return [dict(row) for row in rows]

    def get_recent_news(self, category: Optional[str] = None, hours: int = 24, limit: int = 100) -> list[dict]:
        """Get recent news items."""
        with self._get_connection() as conn:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            query = "SELECT * FROM news WHERE crawled_at > ?"
            params = [cutoff]

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY crawled_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_unacknowledged_signals(self, severity: Optional[str] = None) -> list[dict]:
        """Get signals that haven't been acknowledged."""
        with self._get_connection() as conn:
            query = "SELECT * FROM signals WHERE acknowledged = 0"
            params = []

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            query += " ORDER BY created_at DESC"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def acknowledge_signal(self, signal_id: int):
        """Mark a signal as acknowledged."""
        with self._get_connection() as conn:
            conn.execute("UPDATE signals SET acknowledged = 1 WHERE id = ?", (signal_id,))

    def get_crawl_stats(self, hours: int = 24) -> dict:
        """Get crawl statistics."""
        with self._get_connection() as conn:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            stats = {}
            rows = conn.execute("""
                SELECT source,
                       COUNT(*) as crawl_count,
                       SUM(items_count) as total_items,
                       SUM(error_count) as total_errors,
                       MAX(completed_at) as last_crawl
                FROM crawl_log
                WHERE started_at > ?
                GROUP BY source
            """, (cutoff,)).fetchall()

            for row in rows:
                stats[row["source"]] = dict(row)

            return stats

    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days."""
        with self._get_connection() as conn:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            conn.execute("DELETE FROM price_history WHERE recorded_at < ?", (cutoff,))
            conn.execute("DELETE FROM news WHERE crawled_at < ?", (cutoff,))
            conn.execute("DELETE FROM crawl_log WHERE started_at < ?", (cutoff,))
            conn.execute("DELETE FROM signals WHERE created_at < ? AND acknowledged = 1", (cutoff,))

            logger.info(f"Cleaned up data older than {days} days")
