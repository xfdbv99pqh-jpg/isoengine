#!/usr/bin/env python3
"""
Basic usage example for the Kalshi Web Crawler.

This script demonstrates how to:
1. Run crawlers manually
2. Query stored data
3. Generate and view signals
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from kalshi_crawler import CrawlerConfig, CrawlerRunner


def main():
    print("=" * 60)
    print("Kalshi Web Crawler - Basic Usage Example")
    print("=" * 60)

    # Create configuration
    # API keys are read from environment variables by default
    config = CrawlerConfig(
        db_path=Path("./example_data.db"),
        use_kalshi_demo=True,  # Use demo API for testing
    )

    # Check for missing configuration
    warnings = config.validate()
    if warnings:
        print("\nConfiguration warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    # Create the runner
    runner = CrawlerRunner(config)

    # Show initial status
    print("\n1. Crawler Status:")
    print("-" * 40)
    for name, crawler in runner.crawlers.items():
        status = "✓" if crawler.is_available() else "✗"
        print(f"  {status} {name}")

    # Run crawlers that don't require authentication first
    print("\n2. Running Available Crawlers...")
    print("-" * 40)

    # Run individual crawlers
    sources_to_run = ["polymarket", "rss"]  # These don't require API keys

    for source in sources_to_run:
        print(f"\n  Crawling {source}...")
        results = runner.run_crawler(source)
        print(f"    Retrieved {len(results)} items")

    # If FRED key is available, run it too
    if runner.crawlers["fred"].is_available():
        print("\n  Crawling FRED...")
        results = runner.run_crawler("fred")
        print(f"    Retrieved {len(results)} indicators")

    # Query some data
    print("\n3. Sample Data from Database:")
    print("-" * 40)

    # Latest markets from Polymarket
    markets = runner.db.get_latest_markets(source="polymarket", limit=5)
    if markets:
        print("\n  Top Polymarket Markets:")
        for m in markets:
            import json
            data = json.loads(m.get("data", "{}"))
            question = data.get("question", "")[:60]
            print(f"    - {question}...")

    # Recent news
    news = runner.db.get_recent_news(limit=5)
    if news:
        print("\n  Recent News:")
        for n in news:
            title = n.get("title", "")[:60]
            print(f"    - [{n.get('source')}] {title}...")

    # Economic indicators (if available)
    indicators = runner.db.get_latest_indicators()
    if indicators:
        print("\n  Economic Indicators:")
        for ind in indicators[:5]:
            print(f"    - {ind.get('series_id')}: {ind.get('value')} ({ind.get('value_date')})")

    # Run signal analysis
    print("\n4. Signal Analysis:")
    print("-" * 40)
    signals = runner.analyzer.analyze_all()
    if signals:
        print(f"\n  Generated {len(signals)} signals:")
        for s in signals[:10]:  # Show first 10
            print(f"    [{s.severity}] {s.signal_type}: {s.message[:60]}...")
    else:
        print("  No signals detected (need more data)")

    # Get summary
    print("\n5. Summary:")
    print("-" * 40)
    summary = runner.analyzer.get_market_summary()
    print(f"  Polymarket markets: {summary['polymarket_markets']}")
    print(f"  FRED indicators: {summary['indicators']}")
    print(f"  Recent news items: {summary['recent_news']}")
    print(f"  Active signals: {summary['unacknowledged_signals']}")

    print("\n" + "=" * 60)
    print("Done! Data stored in: example_data.db")
    print("=" * 60)

    # Interactive prompt
    print("\nWant to explore more? Start the interactive shell:")
    print("  python -m kalshi_crawler.runner --shell")


if __name__ == "__main__":
    main()
