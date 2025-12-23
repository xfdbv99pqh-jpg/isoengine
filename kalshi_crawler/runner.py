"""Main orchestrator for running all crawlers."""

import logging
import time
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

from .config import CrawlerConfig
from .base import BaseCrawler, CrawlResult
from .storage.db import CrawlerDatabase
from .analysis.signals import SignalAnalyzer
from .analysis.strategy import StrategyGenerator
from .sources import (
    KalshiCrawler,
    FredCrawler,
    PolymarketCrawler,
    PollingCrawler,
    RSSCrawler,
    EconomicCalendarCrawler,
    BettingOddsCrawler,
    EarningsCalendarCrawler,
    SocialSentimentCrawler,
)

logger = logging.getLogger(__name__)


class CrawlerRunner:
    """Orchestrates all crawlers with scheduling and monitoring."""

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig.from_env()
        self.db = CrawlerDatabase(self.config.db_path)
        self.analyzer = SignalAnalyzer(self.db)
        self.strategy = StrategyGenerator(self.db)

        # Initialize crawlers
        self.crawlers: dict[str, BaseCrawler] = {}
        self._init_crawlers()

        # Scheduling
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=5)

    def _init_crawlers(self):
        """Initialize all configured crawlers."""
        self.crawlers["kalshi"] = KalshiCrawler(self.config)
        self.crawlers["fred"] = FredCrawler(self.config)
        self.crawlers["polymarket"] = PolymarketCrawler(self.config)
        self.crawlers["polling"] = PollingCrawler(self.config)
        self.crawlers["rss"] = RSSCrawler(self.config)
        self.crawlers["economic_calendar"] = EconomicCalendarCrawler(self.config)
        self.crawlers["betting_odds"] = BettingOddsCrawler(self.config)
        self.crawlers["earnings_calendar"] = EarningsCalendarCrawler(self.config)
        self.crawlers["social_sentiment"] = SocialSentimentCrawler(self.config)

        # Log availability
        for name, crawler in self.crawlers.items():
            status = "available" if crawler.is_available() else "unavailable"
            logger.info(f"Crawler '{name}': {status}")

    def run_crawler(self, name: str) -> list[CrawlResult]:
        """Run a single crawler and store results."""
        if name not in self.crawlers:
            logger.error(f"Unknown crawler: {name}")
            return []

        crawler = self.crawlers[name]
        if not crawler.is_available():
            logger.warning(f"Crawler '{name}' is not available")
            return []

        started_at = datetime.utcnow()
        results = []
        error_count = 0

        try:
            logger.info(f"Starting crawl: {name}")
            results = crawler.crawl()

            # Store results
            self.db.store_results(results)
            logger.info(f"Crawl complete: {name} - {len(results)} items")

        except Exception as e:
            logger.error(f"Crawl failed: {name} - {e}")
            error_count = 1

        finally:
            completed_at = datetime.utcnow()
            self.db.log_crawl(
                source=name,
                started_at=started_at,
                completed_at=completed_at,
                items_count=len(results),
                error_count=error_count,
                status="success" if not error_count else "error",
            )

        return results

    def run_all(self, parallel: bool = True) -> dict[str, list[CrawlResult]]:
        """Run all available crawlers."""
        all_results = {}

        if parallel:
            futures = {}
            for name, crawler in self.crawlers.items():
                if crawler.is_available():
                    future = self.executor.submit(self.run_crawler, name)
                    futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    all_results[name] = future.result()
                except Exception as e:
                    logger.error(f"Crawler {name} failed: {e}")
                    all_results[name] = []
        else:
            for name in self.crawlers:
                all_results[name] = self.run_crawler(name)

        # Run signal analysis after crawling
        signals = self.analyzer.analyze_all()
        if signals:
            logger.info(f"Generated {len(signals)} signals")
            for signal in signals:
                if signal.severity in ("warning", "alert"):
                    logger.warning(str(signal))

        return all_results

    def run_scheduled(self):
        """Run crawlers on a schedule."""
        # Real-time sources (every 5 minutes)
        schedule.every(5).minutes.do(lambda: self.run_crawler("kalshi"))
        schedule.every(5).minutes.do(lambda: self.run_crawler("polymarket"))

        # News (every 15 minutes)
        schedule.every(15).minutes.do(lambda: self.run_crawler("rss"))

        # Social sentiment (every 15 minutes)
        schedule.every(15).minutes.do(lambda: self.run_crawler("social_sentiment"))

        # Economic data (hourly, since it doesn't update that often)
        schedule.every(1).hours.do(lambda: self.run_crawler("fred"))

        # Calendars (every 4 hours - these don't change frequently)
        schedule.every(4).hours.do(lambda: self.run_crawler("economic_calendar"))
        schedule.every(4).hours.do(lambda: self.run_crawler("earnings_calendar"))

        # Betting odds (every 30 minutes)
        schedule.every(30).minutes.do(lambda: self.run_crawler("betting_odds"))

        # Polling (every 2 hours)
        schedule.every(2).hours.do(lambda: self.run_crawler("polling"))

        # Signal analysis (every 10 minutes)
        schedule.every(10).minutes.do(self.analyzer.analyze_all)

        # Daily cleanup
        schedule.every().day.at("03:00").do(lambda: self.db.cleanup_old_data(days=30))

        logger.info("Scheduled crawlers started")
        self.running = True

        # Run initial crawl
        self.run_all()

        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop scheduled crawlers."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Crawler runner stopped")

    def get_status(self) -> dict:
        """Get status of all crawlers."""
        return {
            "crawlers": {name: crawler.get_status() for name, crawler in self.crawlers.items()},
            "summary": self.analyzer.get_market_summary(),
            "config_warnings": self.config.validate(),
        }

    def interactive_shell(self):
        """Run an interactive shell for querying data."""
        import cmd

        runner = self

        class CrawlerShell(cmd.Cmd):
            intro = "Kalshi Crawler Shell. Type 'help' for commands."
            prompt = "crawler> "

            def do_crawl(self, arg):
                """Crawl a source: crawl <source|all>"""
                if arg == "all":
                    runner.run_all()
                elif arg in runner.crawlers:
                    runner.run_crawler(arg)
                else:
                    print(f"Unknown source. Available: {', '.join(runner.crawlers.keys())}")

            def do_status(self, arg):
                """Show crawler status"""
                import json
                print(json.dumps(runner.get_status(), indent=2, default=str))

            def do_markets(self, arg):
                """Show latest markets: markets [source] [category]"""
                parts = arg.split()
                source = parts[0] if parts else None
                category = parts[1] if len(parts) > 1 else None
                markets = runner.db.get_latest_markets(source=source, category=category, limit=20)
                for m in markets:
                    print(f"  {m.get('ticker')}: {m.get('title', '')[:60]}")

            def do_news(self, arg):
                """Show recent news: news [category]"""
                category = arg if arg else None
                news = runner.db.get_recent_news(category=category, limit=20)
                for n in news:
                    print(f"  [{n.get('source')}] {n.get('title', '')[:70]}")

            def do_signals(self, arg):
                """Show unacknowledged signals"""
                signals = runner.db.get_unacknowledged_signals()
                for s in signals:
                    print(f"  [{s.get('severity')}] {s.get('signal_type')}: {s.get('message')}")

            def do_indicators(self, arg):
                """Show latest economic indicators"""
                indicators = runner.db.get_latest_indicators()
                for i in indicators:
                    print(f"  {i.get('series_id')}: {i.get('value')} ({i.get('value_date')})")

            def do_analyze(self, arg):
                """Run signal analysis"""
                signals = runner.analyzer.analyze_all()
                print(f"Generated {len(signals)} signals")
                for s in signals:
                    print(f"  {s}")

            def do_calendar(self, arg):
                """Show upcoming economic events: calendar [days]"""
                days = int(arg) if arg else 7
                crawler = runner.crawlers.get("economic_calendar")
                if crawler:
                    events = crawler.get_upcoming_events(days)
                    print(f"\nUpcoming events (next {days} days):")
                    for e in events:
                        print(f"  {e.get('date')}: {e.get('name')} ({e.get('type')})")

            def do_earnings(self, arg):
                """Show upcoming earnings: earnings [days]"""
                days = int(arg) if arg else 7
                crawler = runner.crawlers.get("earnings_calendar")
                if crawler:
                    earnings = crawler.get_upcoming_earnings(days)
                    print(f"\nUpcoming earnings (next {days} days):")
                    for e in earnings:
                        print(f"  {e.get('earnings_date')}: {e.get('symbol')} - {e.get('company', 'N/A')}")

            def do_sentiment(self, arg):
                """Show social sentiment summary"""
                crawler = runner.crawlers.get("social_sentiment")
                if crawler:
                    summary = crawler.get_sentiment_summary()
                    print(f"\nSocial Sentiment Summary:")
                    print(f"Overall: {summary['overall']}")
                    for cat, scores in summary.get('by_category', {}).items():
                        print(f"  {cat}: {scores}")

            def do_odds(self, arg):
                """Show election betting odds"""
                crawler = runner.crawlers.get("betting_odds")
                if crawler:
                    consensus = crawler.get_election_consensus()
                    print(f"\nBetting Odds Consensus:")
                    for candidate, data in consensus.get('candidates', {}).items():
                        prob = data.get('avg_probability', 0) * 100
                        print(f"  {candidate}: {prob:.1f}%")

            def do_strategy(self, arg):
                """Generate investment strategy report: strategy [picks]"""
                if arg == "picks":
                    picks = runner.strategy.get_top_picks(10)
                    print(f"\nTOP {len(picks)} PICKS:")
                    print("-" * 50)
                    for pick in picks:
                        print(pick)
                        print()
                else:
                    print(runner.strategy.print_strategy_report())

            def do_picks(self, arg):
                """Show top trading picks: picks [number]"""
                n = int(arg) if arg else 5
                picks = runner.strategy.get_top_picks(n)
                print(f"\n{'='*60}")
                print(f"TOP {len(picks)} KALSHI PICKS")
                print(f"{'='*60}\n")
                for i, pick in enumerate(picks, 1):
                    conf = {"HIGH": "★★★", "MEDIUM": "★★☆", "LOW": "★☆☆"}.get(pick.confidence, "?")
                    action = {"BUY_YES": "🟢 YES", "BUY_NO": "🔴 NO"}.get(pick.action, pick.action)
                    print(f"{i}. {conf} {action} @ ${pick.current_price:.2f}" if pick.current_price else f"{i}. {conf} {action}")
                    print(f"   {pick.ticker}")
                    print(f"   {pick.title[:55]}...")
                    for r in pick.reasoning[:2]:
                        print(f"   → {r}")
                    print()

            def do_quit(self, arg):
                """Exit the shell"""
                return True

            do_exit = do_quit
            do_q = do_quit

        CrawlerShell().cmdloop()


def main():
    """Main entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Kalshi Web Crawler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--source", type=str, help="Run specific source only")
    parser.add_argument("--shell", action="store_true", help="Start interactive shell")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--strategy", action="store_true", help="Generate investment strategy report")
    parser.add_argument("--picks", type=int, metavar="N", help="Show top N trading picks")
    parser.add_argument("--days-min", type=int, default=1, help="Minimum days to expiry (default: 1)")
    parser.add_argument("--days-max", type=int, default=14, help="Maximum days to expiry (default: 14)")
    args = parser.parse_args()

    runner = CrawlerRunner()

    if args.status:
        import json
        print(json.dumps(runner.get_status(), indent=2, default=str))
    elif args.strategy:
        print(runner.strategy.print_strategy_report())
    elif args.picks:
        days_min = args.days_min
        days_max = args.days_max
        picks = runner.strategy.get_top_picks(args.picks, days_min=days_min, days_max=days_max)
        print(f"\n{'='*60}")
        print(f"TOP {len(picks)} KALSHI PICKS (settling in {days_min}-{days_max} days)")
        print(f"{'='*60}\n")
        if not picks:
            print("No picks found matching criteria.")
            print("Try: --days-max 30 for longer timeframe")
            print("Or run --once first to crawl fresh data")
        for i, pick in enumerate(picks, 1):
            conf = {"HIGH": "★★★", "MEDIUM": "★★☆", "LOW": "★☆☆"}.get(pick.confidence, "?")
            action = {"BUY_YES": "🟢 YES", "BUY_NO": "🔴 NO"}.get(pick.action, pick.action)
            price_str = f" @ ${pick.current_price:.2f}" if pick.current_price else ""
            days_str = f" ({pick.signal.days_to_expiry}d)" if pick.signal and pick.signal.days_to_expiry else ""
            print(f"{i}. {conf} {action}{price_str}{days_str}")
            print(f"   {pick.ticker}")
            print(f"   {pick.title[:55]}...")
            for r in pick.reasoning[:3]:
                print(f"   → {r}")
            print()
    elif args.shell:
        runner.interactive_shell()
    elif args.source:
        runner.run_crawler(args.source)
    elif args.once:
        runner.run_all()
    else:
        try:
            runner.run_scheduled()
        except KeyboardInterrupt:
            runner.stop()


if __name__ == "__main__":
    main()
