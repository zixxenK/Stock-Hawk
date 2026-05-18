# AGENTS.md

## Purpose
This repository implements the Flippy Engine: a production-grade, alternative-data-driven quantitative trading system with a Gymnasium/PPO training environment.

## What agents should know
- This is not a prototype repo. Every implementation must be complete and production-ready.
- Do not introduce or leave any `pass`, `...`, `# TODO`, stub, or partial implementation in structural code.
- Use Python 3.10+ type annotations everywhere.
- Use Pydantic v2+ schemas for raw data boundaries and validation.
- Preserve module separation: ingestion, persistence, signal construction, RL environment, API, and execution logic are distinct responsibilities.

## Key architectural constraints
- `rl_insider_trader.py` is the core training environment.
  - It must subclass `gym.Env` and implement `reset(seed=seed)` with `super().reset(seed=seed)`.
  - `step(action)` must return `(observation, reward, terminated, truncated, info)`.
  - `terminated` must only signal natural episode completion.
  - `truncated` must signal capital-ruin or other artificial cutoff.
  - `_get_obs()` must return `np.array(..., dtype=np.float32)` with shape `(11,)`.

- Observation state must be strictly bounded and causal.
  - Log return and technical features must be backward-looking only.
  - Use `adjust=False` for EWMA-based indicators where applicable.
  - No future-peeking rolling windows or center-aligned windows.

- Reward shaping must follow the contract:
  - `R_t = R_pnl - churn + alignment - drawdown`
  - Apply symmetric clipping to `[-5.0, 5.0]`.
  - Penalize churn, reward alignment to insider/congress signals, and penalize large drawdowns.

- Alternative data ingestion must enforce latency and causality.
  - Public disclosure must be visible only when `D_disclosure <= D_sim_step`.
  - Emulate reporting delays for congressional and insider filings.
  - Post-market events after 20:00 UTC are only available on the next trading day.

- Database persistence must use SQLite WAL mode.
  - Any database manager or adapter must execute `PRAGMA journal_mode=WAL` on initialization.
  - This is required to support concurrent reads during active ingestion.

- Execution and trading logic should treat fees and slippage as first-class costs.
  - Order allocation must deduct friction before share allocation.
  - Enforce a max single-trade exposure of `10%` of current equity.
  - Implement a position-level trailing stop-loss override at `-5%` from entry.

## Important files
- `README.md` — architecture overview and system rationale.
- `rl_insider_trader.py` — custom Gymnasium environment and PPO training pipeline.
- `database/db_manager.py` — SQLite persistence layer.
- `data_sources/scraper_manager.py` — ingestion orchestration.
- `api/fast_api_app.py` — REST endpoints for analysis and backtest.
- `main.py` — golden pipeline wiring.
- `processing/sentiment_processor.py` — NLP sentiment processing.
- `intelligence/agent.py` — reinforcement learning and signal logic.

## Best practices for AI code generation
- Prefer existing repository structure and file boundaries over creating new monolithic files.
- Link to `README.md` for design rationale rather than duplicating large conceptual text.
- Use the existing data model conventions in `database/` and `config/`.
- Validate all input data and output shapes explicitly.

## When in doubt
- If a requested change affects the trading environment, consult `rl_insider_trader.py` and the reward/observation contract.
- If it affects data ingestion, preserve the raw data lake pattern and do not write parsed entities before storing raw bytes.
- If it affects persistence, ensure WAL mode and safe concurrency.

## Recommended next customization
- Create a dedicated skill for `flipper_infrastructure` or broker execution if the repo later expands live ingestion and Alpaca integration.
