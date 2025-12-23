# Kalshi Web Crawler

A modular data aggregation system for informed prediction market trading. Collects data from multiple sources, stores it locally, and generates trading signals.

## Features

- **Multi-source data collection**
  - Kalshi API (prediction markets)
  - FRED API (economic indicators)
  - Polymarket API (cross-market comparison)
  - Polling aggregators (538, RealClearPolitics)
  - RSS feeds (news sources)

- **Rate-limited, respectful crawling** with automatic retry logic
- **SQLite storage** for time-series data
- **Signal extraction** for trading opportunities
- **Cross-market arbitrage detection**
- **Scheduled background crawling**

## Quick Start

### 1. Install dependencies

```bash
pip install requests beautifulsoup4 schedule lxml
```

### 2. Set up API keys (optional but recommended)

```bash
# Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="your-fred-api-key"

# Kalshi API (for authenticated endpoints)
export KALSHI_API_KEY="your-kalshi-email"
export KALSHI_API_SECRET="your-kalshi-password"
```

### 3. Run a single crawl

```python
from kalshi_crawler import CrawlerRunner

runner = CrawlerRunner()
results = runner.run_all()

# Check status
print(runner.get_status())
```

### 4. Run scheduled crawling

```python
runner = CrawlerRunner()
runner.run_scheduled()  # Ctrl+C to stop
```

### 5. Use the interactive shell

```bash
python -m kalshi_crawler.runner --shell
```

```
crawler> crawl all
crawler> markets kalshi economics
crawler> news politics
crawler> signals
crawler> status
crawler> quit
```

## Configuration

Configure via environment variables or by passing a `CrawlerConfig` object:

```python
from kalshi_crawler import CrawlerConfig, CrawlerRunner

config = CrawlerConfig(
    fred_api_key="your-key",
    db_path=Path("./my_data.db"),
    use_kalshi_demo=True,  # Use demo API for testing
)

runner = CrawlerRunner(config)
```

### Available Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `fred_api_key` | env `FRED_API_KEY` | FRED API key |
| `kalshi_api_key` | env `KALSHI_API_KEY` | Kalshi email |
| `kalshi_api_secret` | env `KALSHI_API_SECRET` | Kalshi password |
| `db_path` | `./kalshi_data.db` | SQLite database path |
| `use_kalshi_demo` | `True` | Use Kalshi demo API |
| `default_rate_limit` | 30 | Requests per minute |

## Data Sources

### Kalshi (Prediction Markets)
- All open markets with prices, volume, and order book
- Categorized by: economics, politics, tech, weather, other
- Update frequency: Real-time (every 5 min in scheduled mode)

### FRED (Federal Reserve Economic Data)
- Key indicators: Unemployment, CPI, GDP, Fed Funds Rate, Treasury yields
- Automatic change calculations (period-over-period, year-over-year)
- Update frequency: Hourly (data updates vary by series)

### Polymarket
- Active markets for cross-reference with Kalshi
- Enables arbitrage detection between platforms
- Update frequency: Real-time (every 5 min)

### Polling
- 538 and RealClearPolitics polling averages
- Presidential approval, generic ballot, election forecasts
- Update frequency: Every 2 hours

### RSS Feeds
- AP Politics, Reuters, Fed News, HackerNews, TechCrunch
- Keyword detection for market-moving news
- Update frequency: Every 15 minutes

## Signal Types

The analyzer generates signals for:

1. **Price Movements** - Significant price changes (>10% in 24h)
2. **Cross-Market Arbitrage** - Price discrepancies between Kalshi and Polymarket
3. **Indicator Surprises** - Economic data outside normal ranges
4. **News Alerts** - Keywords like "breaking", "recession", "fed rate"

```python
# Get unacknowledged signals
signals = runner.db.get_unacknowledged_signals()
for s in signals:
    print(f"[{s['severity']}] {s['message']}")
    runner.db.acknowledge_signal(s['id'])
```

## Database Schema

Data is stored in SQLite with these main tables:

- `markets` - Market snapshots (Kalshi, Polymarket)
- `price_history` - Time-series price data
- `indicators` - Economic indicator values
- `news` - RSS feed items
- `polls` - Polling data
- `signals` - Generated trading signals
- `crawl_log` - Crawl execution history

## Adding Custom Sources

Extend `BaseCrawler` to add new sources:

```python
from kalshi_crawler.base import BaseCrawler, CrawlResult
from datetime import datetime

class MyCustomCrawler(BaseCrawler):
    def __init__(self, config):
        super().__init__(
            name="mycrawler",
            rate_limit=30,
            timeout=30,
        )

    def is_available(self) -> bool:
        return True

    def crawl(self) -> list[CrawlResult]:
        # Your crawl logic here
        response = self.get("https://api.example.com/data")
        data = response.json()

        return [CrawlResult(
            source="mycrawler",
            category="custom",
            data_type="market",
            timestamp=datetime.utcnow(),
            data=data,
        )]
```

## Adding Custom RSS Feeds

```python
config = CrawlerConfig()
config.rss_feeds.append({
    "name": "My Feed",
    "url": "https://example.com/rss",
    "category": "custom",
})
```

## Command Line Usage

```bash
# Run all crawlers once
python -m kalshi_crawler.runner --once

# Run specific source
python -m kalshi_crawler.runner --source kalshi

# Show status
python -m kalshi_crawler.runner --status

# Interactive shell
python -m kalshi_crawler.runner --shell

# Run scheduled (continuous)
python -m kalshi_crawler.runner
```

## Example: Finding Arbitrage Opportunities

```python
from kalshi_crawler import CrawlerRunner

runner = CrawlerRunner()
runner.run_all()

# Compare Kalshi prices to indicators
comparisons = runner.analyzer.compare_kalshi_to_indicators()
for comp in comparisons:
    market = comp['market']
    indicators = comp['indicators']
    print(f"\n{market['ticker']}: {market['title']}")
    print(f"  Kalshi Yes Price: {market['yes_price']}")
    for ind in indicators:
        print(f"  {ind['name']}: {ind['value']} ({ind['date']})")

# Check for cross-market price differences
signals = runner.analyzer.detect_cross_market_arbitrage(threshold=0.05)
for signal in signals:
    print(signal)
```

## Expanding the System

As suggested in the overview, you can expand to 20+ sources:

| Category | Additional Sources to Add |
|----------|--------------------------|
| Economics | CME FedWatch, Treasury yields, ISM data |
| Politics | Congress.gov, Ballotpedia, White House briefings |
| Tech | SEC EDGAR, FTC/DOJ newsrooms, Company IR pages |
| Cross-market | PredictIt, Metaculus |

Each can be added by creating a new crawler class following the `BaseCrawler` pattern.

## License

MIT
