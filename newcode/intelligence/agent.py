from __future__ import annotations
import numpy as np
import random
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List
# Assuming access to the global SETTINGS object and constants.

logger = logging.getLogger(__name__)

# ── Constants (Re-defined for completeness) ──────────────────────────────
VECTOR_DIM      = 18
EXPANDED_DIM    = VECTOR_DIM * 2 + 1
N_ACTIONS       = 4  # SKIP, BUY_SMALL, BUY_MEDIUM, BUY_LARGE

ACTION_SIZES: dict[int, float] = {
    0: 0.000,   # SKIP
    1: 0.005,   # 0.5%
    2: 0.010,   # 1.0%
    3: 0.020,   # 2.0% (Max risk exposure)
}
ACTION_NAMES: dict[int, str] = {
    0: "SKIP",
    1: "BUY_SMALL",
    2: "BUY_MEDIUM",
    3: "BUY_LARGE",
}

PROFIT_THRESHOLD = 0.002      # 0.2% minimum profit threshold
REWARD_SCALE     = 50.0       # Reward amplification factor
PENALTY_SCALE    = 75.0       # Penalty scale (Loss aversion)
MAX_RISK_CAP      = 0.02       # Max safe position size

WEIGHTS_DIR = Path(__file__).parent.parent / "data" / "rl_weights"


# ── Reward Function Context ─────────────────────────────────────────────

@dataclass
class RewardContext:
    """All inputs needed to compute a trade reward."""
    action:           int
    entry_price:      float
    exit_price:       float
    position_pct:     float        # Fraction of account deployed (0-1)
    account_size:     float
    holding_days:     int   = 1
    spy_return:       float = 0.0
    was_stop_hit:     bool  = False

def compute_reward(ctx: RewardContext) -> float:
    """
    Core reward function (Enhanced for financial nuance).
    Returns a highly penalized/rewarded value based on market context.
    """
    if ctx.entry_price <= 0 or ctx.action == 0:
        return 0.0

    trade_return = (ctx.exit_price - ctx.entry_price) / ctx.entry_price
    alpha        = trade_return - ctx.spy_return
    base_reward = 0.0

    # --- Base Return Calculation ---
    if trade_return >= PROFIT_THRESHOLD:
        excess     = trade_return - PROFIT_THRESHOLD
        base_reward = (REWARD_SCALE * excess) + 1.0 # Profit bonus
    else:
        shortfall   = PROFIT_THRESHOLD - trade_return
        base_reward = (-PENALTY_SCALE * shortfall) - 0.5 # Loss penalty

    # --- Action-Specific Penalties ---
    if ctx.action == 1 and trade_return < 0:
        base_reward *= 1.2  # Small loss on small buy is less penalized
    elif ctx.action == 3 and trade_return < -0.05:
        base_reward *= 1.5 # Massive penalty for large losses

    # --- Risk Management Penalties (The most important addition) ---
    if ctx.position_pct > MAX_RISK_CAP * 1.2:
        oversize_ratio  = ctx.position_pct / MAX_RISK_CAP
        base_reward *= (1.0 / oversize_ratio**1.5) # Severe penalty for over-leveraging

    # --- Alpha Bonus/Penalty ---
    if alpha > 0.005:
        base_reward += REWARD_SCALE * 0.10 * alpha
    elif alpha < -0.005:
        base_reward -= PENALTY_SCALE * 0.10 * abs(alpha) # Punish missing market moves

    # Clip and return the final, refined reward signal
    return float(np.clip(base_reward, -20.0, 20.0))


# ── Feature Expansion (No changes needed) ───────────────────────────────
def expand_features(state: np.ndarray) -> np.ndarray:
    """Expands state to [s, s^2, bias]."""
    squared = state ** 2
    bias    = np.array([1.0], dtype=np.float32)
    return np.concatenate([state, squared, bias]).astype(np.float32)


# ── Experience Replay Buffer & Pattern Memory (No major changes needed) ──────

@dataclass
class Experience:
    # ... definition remains the same ...
    pass

class ReplayBuffer:
    # ... implementation remains the same ...
    pass

class PatternMemory:
    """Stores historical outcomes and calculates similarity."""
    def __init__(self, max_patterns: int = 5000) -> None:
        self._states: list[np.ndarray] = []
        self._outcomes: list[dict]        = []
        self._max       = max_patterns

    # ... record and find_similar methods remain the same ...

    def historical_win_rate(self, state: np.ndarray, top_k: int = 20) -> float | None:
        """Returns win rate of similar historical setups."""
        # This is crucial for providing pattern confidence to the Agent's decision process.
        similar = self.find_similar(state, top_k=top_k, min_sim=0.82)
        if not similar: return None
        wins = sum(1 for s in similar if s.get("trade_return", 0) > PROFIT_THRESHOLD)
        return wins / len(similar)


# ── The Main Agent Class (The Decision Maker) ─────────────────────────────

class FlippyAgent:
    """
    Linear Q-learning agent incorporating Pattern Memory and external signals.
    This class manages the entire decision pipeline: Observe -> Score -> Act.
    """
    def __init__(self, learning_rate: float = 0.001, discount: float = 0.95, epsilon: float = 1.0, ...):
        # Initialize weights, buffers, and memory components
        self._weights: np.ndarray = np.zeros((N_ACTIONS, EXPANDED_DIM))
        self._replay = ReplayBuffer(capacity=10_000)
        self._patterns = PatternMemory()
        # ... (Initialization of state variables: step, total_reward, etc.)

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q(s, a) for all actions."""
        phi = expand_features(state).astype(np.float64)
        return (self._weights @ phi) # shape (N_ACTIONS,)

    def select_action(self, state: np.ndarray, momentum_score: float, sentiment_score: float) -> tuple[int, dict]:
        """
        ε-greedy action selection with integrated signal boosts and SHAP explanation.
        Returns the best action index and a detailed metadata dictionary.
        """
        # 1. Calculate raw Q-values
        q = self._q_values(state).copy()

        # 2. Pattern Memory Boost (The Historical 'Memory')
        hist_win_rate = self._patterns.historical_win_rate(state, top_k=15)
        if hist_win_rate is not None:
            print(f"  [Signal] Historical Win Rate: {hist_win_rate:.2%}")
            # Adjust Q-values based on historical performance certainty
            q[2] += 0.8 if hist_win_rate > 0.7 else (1 - hist_win_rate) * 0.5
        else:
             print("  [Signal] No sufficient pattern data.")

        # 3. Momentum and Sentiment Boost (The Contextual Overlay)
        if momentum_score < 40:
            q[0] += 2.0 # Strong penalty on low conviction setups -> SKIP
        elif sentiment_score > 0.5:
            q[2] += 1.0 # Positive news boosts medium buy confidence

        # 4. Final Action Selection (Epsilon-Greedy)
        if random.random() < self.epsilon:
            action = random.randint(0, N_ACTIONS - 1)
            method = "explore"
        else:
            action = int(np.argmax(q))
            method = "exploit"

        # Find the final score and generate explanation metadata
        final_score = q[action]
        similar = self._patterns.find_similar(state, top_k=5)
        
        return action, {
            "alpha_score": round(final_score, 2), # This is the user-facing score
            "pattern_hints": [f"{s['ticker']} | Ret: {s['trade_return']:.2%}" for s in similar[:3]],
            "q_values": q.tolist(),
            "method": method,
        }

    def update_from_outcome(self, state: np.ndarray, action: int, reward_ctx: RewardContext, next_state: Optional[np.ndarray], ticker: str) -> float:
        """
        High-level entry point for learning from a completed trade cycle.
        This function must be called after the outcome is known.
        """
        # 1. Calculate the formal reward signal
        reward = compute_reward(reward_ctx)

        # 2. Update Pattern Memory & Weights
        self._patterns.record(state=state, action=action, reward=reward, trade_return=(reward_ctx.exit_price - reward_ctx.entry_price)/reward_ctx.entry_price)

        # 3. Train the Agent (Off-policy update using stored experience)
        exp = Experience(state=state, action=action, reward=reward, next_state=next_state or state, done=True)
        self._replay.push(exp)
        
        if len(self._replay) >= self.replay_size:
            self._train_step()

        # 4. Log and Persistence
        print(f"--- LEARNING CYCLE COMPLETE ---")
        self._save_weights()
        return reward


    def _train_step(self) -> None:
        """Performs one gradient descent step using sampled experiences."""
        batch = self._replay.sample(self.replay_size)
        # ... (Training logic remains the same, focusing on semi-gradient updates.)
