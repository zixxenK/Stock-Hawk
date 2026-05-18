# Copilot Instructions

This repository implements the Flippy Engine, an institutional-grade alternative data trading system with a Gymnasium/PPO reinforcement learning workflow.

## Use these guides
- Primary repo guidance: `AGENTS.md`
- Live ingestion and broker execution skill: `SKILL.md`
- System design rationale: `README.md`

## Key expectations
- No `pass`, `...`, or `# TODO` in structural code.
- Use Python 3.10+ type annotations and Pydantic v2+ models for raw data boundaries.
- Preserve module separation: ingestion, persistence, signal processing, RL environment, API, and execution must remain distinct.
- Enforce SQLite WAL mode in database adapters with `PRAGMA journal_mode=WAL`.
- RL environment observations must be `np.array(..., dtype=np.float32)` with shape `(11,)`.

## When editing code
- Consult `rl_insider_trader.py` for the Gymnasium contract and reward shaping rules.
- Consult `database/db_manager.py` for persistence patterns.
- Consult `data_sources/scraper_manager.py` for ingestion orchestration.
- Do not copy large documentation into code; link to `README.md` instead.
