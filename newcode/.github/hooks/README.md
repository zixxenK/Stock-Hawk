# Flippy Engine Agent Hooks

This directory contains workspace hooks for Flippy Engine agent behavior and policy enforcement.

## `flippy_agent_rules.json`

This hook enforces repository-specific constraints before agents take action.

### What it does
- Injects high-level repo guidance from `AGENTS.md` and `SKILL.md`.
- Warns when code contains placeholder or incomplete patterns such as `pass`, `...`, or `# TODO`.
- Warns if SQLite persistence does not include `PRAGMA journal_mode=WAL`.
- Warns if RL observation vectors are not explicitly cast to `np.float32`.
- Blocks requests that rely on mock or fake data instead of real hardened sources.

### Trigger events
- `SessionStart` — establish context at the beginning of a coding session.
- `PreToolUse` — validate agent behavior before tools are executed.

## How to use
- Add or update hooks in this directory to enforce new repository policies.
- Keep hook rules aligned with `AGENTS.md` and `SKILL.md` so agents follow the same architecture expectations.

## Why it exists
This hook ensures that any AI-driven changes remain production-grade, modular, and consistent with the Flippy Engine design contract. It is especially useful for preventing incomplete or non-compliant code from being introduced during automated updates.
