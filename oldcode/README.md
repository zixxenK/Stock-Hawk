# Momentum Scanner

A long-term momentum stock scanner for manual trade execution.
Scans the S&P 500 + Nasdaq 100 and ranks stocks by a composite
momentum score built on eight academically-backed factors.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full scan (first run downloads data — takes ~3-5 min)
python run.py

# 3. Show detailed cards + position sizing for top 5
python run.py --account 50000 --detail 5

# 4. Export to CSV
python run.py --csv
```

---

## How the Score Works

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| 12-1 Month Momentum | 25% | Return from 12 months ago to 1 month ago (skips last month to avoid short-term reversal) |
| Relative Strength vs SPY | 20% | How much the stock beat the S&P 500 |
| 52-Week High Proximity | 15% | Stocks near their 52-week high tend to keep rising |
| Sharpe Momentum | 10% | Consistent, smooth uptrend (not just a single spike) |
| Golden Cross / Trend | 10% | 50 SMA above 200 SMA + price above 200 SMA |
| ADX Trend Strength | 8% | Confirms there's actually a trend in the price |
| Volume Trend | 7% | Expanding volume = institutional accumulation |
| RSI Zone | 5% | Ideal entry is RSI 55-70 (momentum, not overbought) |

**Score interpretation:**
- 90-100 (A+): Elite momentum — high conviction
- 80-89 (A): Strong setup
- 70-79 (B+): Good setup — worth watching
- 60-69 (B): Decent — watch for catalyst
- < 60: Weak or no momentum

---

## All Command-Line Options

```
python run.py [options]

Universe:
  --tickers AAPL MSFT NVDA    Scan specific tickers only
  --nasdaq-only               Scan Nasdaq 100 only (faster, ~200 tickers)

Filters:
  --top 30                    Show top N results (default: 30)
  --min-score 55              Minimum score to show (default: 55)
  --min-price 5               Skip stocks below $5 (default: $5)
  --max-price 500             Optional maximum price filter

Risk & Position Sizing:
  --account 50000             Your account size in dollars
  --risk 0.01                 Risk per trade (default: 1% of account)

Output:
  --csv                       Export to ./reports/scan_YYYYMMDD_HHMM.csv
  --detail 5                  Show detailed card for top 5 results
  --benchmark QQQ             Change benchmark (default: SPY)

Misc:
  --no-cache                  Force re-download of stock universe
  -v, --verbose               Debug logging
```

---

## Position Sizing

When you pass `--account` and `--detail`, each detailed card shows:

```
Entry:     $150.00
Stop:      $143.00   (2× ATR below entry)
Target 1R: $157.00   (break even / trail stop here)
Target 2R: $164.00   ← recommended minimum exit
Target 3R: $171.00   (let runners reach this)
Shares:    33        ($4,950 deployed)
$ at risk: $231      (0.46% of $50k account)
```

**The 1% rule:** Each trade risks ≤ 1% of your account.
If you have 10 positions open, your total open risk is ~10%.

---

## Trading Workflow

1. **Run scan** each week (Sunday evening or Monday pre-market)
2. **Review top 20** — look for A+ and A grades
3. **Check chart manually** — confirm the setup looks clean
4. **Size the position** using `--account --detail`
5. **Place limit order** at or slightly above current ask
6. **Set stop loss** at the price shown (hard stop, not mental)
7. **Trail stop** to 1R once the trade is up 2R

### What to look for on the chart
- Clean uptrend with higher highs / higher lows
- Recent pullback to the 21 EMA or 50 SMA (lower-risk entry)
- Volume expanding on up-days, contracting on pullbacks
- No earnings within 2 weeks (earnings = binary risk)

---

## Caching

Price data is cached in `.cache/prices/` as Parquet files.
Cache expires after **4 hours** during market hours.
Universe list (S&P 500 / Nasdaq 100 tickers) is cached for **7 days**.

To force a full re-download:
```bash
python run.py --no-cache
rm -rf .cache/prices/
```

---

## File Structure

```
momentum_scanner/
├── run.py                  ← Entry point
├── requirements.txt
├── README.md
├── scanner/
│   ├── __init__.py
│   ├── universe.py         ← Stock universe (S&P 500 + NDX 100)
│   ├── data.py             ← yfinance fetching + disk caching
│   ├── indicators.py       ← All technical indicators
│   ├── scoring.py          ← Composite score (0-100) + grade
│   ├── risk.py             ← Position sizing + stop loss
│   ├── scanner.py          ← Core scan loop
│   └── display.py          ← Rich terminal output + CSV export
├── .cache/                 ← Auto-created, gitignored
│   ├── universe.json
│   └── prices/
└── reports/                ← Auto-created, CSV exports land here
```

---

## ⚠️ Disclaimer

This tool provides **analysis only** — it does not place trades.
Past momentum does not guarantee future returns.  Always:
- Do your own due diligence on each name
- Check for upcoming earnings before entering
- Never risk more than you can afford to lose
- Consider consulting a registered financial advisor

Momentum strategies have strong historical evidence but can suffer
significant drawdowns during market reversals (e.g. 2022, dot-com bust).
Size positions conservatively and use hard stop losses.
