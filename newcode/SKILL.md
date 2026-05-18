# SKILL.md

## Purpose
This skill helps AI agents implement the Flippy Engine’s live ingestion and broker execution infrastructure in a production-ready way. It is focused on expanding the repository with `flipper_infrastructure` patterns such as raw data lake ingestion, resilient scrapers, causal alternative signal processing, and Alpaca execution controls.

## When to use
- When adding or extending ingestion adapters for Capitol Trades, OpenInsider, news RSS, or options flow feeds.
- When implementing Alpaca paper trading, execution cost models, slippage, order sizing, or risk guardrails.
- When the repository is expanded from offline analysis into a live / near-live trading pipeline.

## What to do
- Preserve module separation: ingestion, persistence, execution, and model signal processing must remain distinct.
- Enforce the raw data lake pattern: save original JSON/HTML responses to `./data/lake/{source}/raw_{timestamp}.json` before parsing.
- Implement user-agent rotation, rate limiting, connection pooling, and exponential backoff with jitter for HTTP ingestion.
- Make all external adapters fault-tolerant and auditable; never write parsed entities until raw bytes are safely stored.
- Use SQLite WAL mode in any database adapter and run `PRAGMA journal_mode=WAL` at connection initialization.
- Use strict Python 3.10+ typing and Pydantic v2+ models for all raw payloads and parsed records.

## Broker execution guidance
- Build Alpaca integration as a separate execution wrapper or module, not inside the RL environment.
- Apply transaction friction before share allocation.
- Compute order capacity using `NotionalLimit = V_t * 0.10` and do not exceed it.
- Simulate a market impact model with `FrictionPct = FlatFeePct + lambda * sqrt(OrderVolume / MarketVolume)`.
- Enforce a position-level trailing stop-loss override at `-5%` from entry price.
- Treat fees and slippage as first-class costs in every portfolio update.

## Important constraints
- No `pass`, `...`, or `# TODO` may remain in structural code.
- Every method must be fully implemented and validated.
- Observation vectors for the RL environment must use `np.array(..., dtype=np.float32)` and shape `(11,)`.
- Data ingestion must enforce latency: filing disclosures after 20:00 UTC are only visible on the next trading day.
- Congressional and insider filings must be delayed in simulation to emulate legal reporting lags.

## Key repo references
- `AGENTS.md` — repo-wide AI agent guidance.
- `README.md` — architecture overview and signal design rationale.
- `rl_insider_trader.py` — custom Gymnasium environment and reward contract.
- `database/db_manager.py` — persistence conventions and WAL requirement.
- `data_sources/scraper_manager.py` — current ingestion orchestration.
- `api/fast_api_app.py` — REST and backtest interface patterns.

## Example prompts
- "Create a new `flipper_infrastructure` module that saves raw HTTP responses, parses Capitol Trades and OpenInsider data, and writes validated records to SQLite WAL." 
- "Extend Alpaca broker execution with a 10% notional cap, Almgren-Chriss slippage, and a 5% trailing stop-loss override."
