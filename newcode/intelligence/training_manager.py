from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

from intelligence.llm_advisor import LLMRecommenderResult, LocalLLMAdvisor

logger = logging.getLogger(__name__)

EnvironmentBuilder = Callable[
    [str, int, Any, Any, Any], tuple[Any, Any, list[dict[str, Any]]]
]


@dataclass
class AutonomousTrainingConfig:
    training_steps: int = 1_000_000
    learning_rate: float = 3e-4
    entropy_coef: float = 0.02
    backtest_days: int = 2520
    strategy_name: str = "PPO_AutoTune"
    min_sharpe_improvement: float = 0.05
    plateau_iterations: int = 2
    max_training_steps: int = 5_000_000
    min_training_steps: int = 50_000
    min_backtest_days: int = 400
    max_backtest_days: int = 5200


@dataclass
class TrainingIterationSummary:
    iteration: int
    metrics: dict[str, float]
    recommendation: LLMRecommenderResult
    config: dict[str, Any]
    completed: bool


@dataclass
class AutonomousTrainingReport:
    ticker: str
    iterations: list[TrainingIterationSummary] = field(default_factory=list)
    best_sharpe: float = 0.0
    final_config: dict[str, Any] = field(default_factory=dict)
    stopped_on_plateau: bool = False
    stopped_reason: str = ""


class AutonomousTrainingManager:
    def __init__(self, db: Any, advisor: LocalLLMAdvisor, env_builder: EnvironmentBuilder) -> None:
        self.db = db
        self.advisor = advisor
        self.env_builder = env_builder

    def run_autonomous_loop(
        self,
        ticker: str,
        initial_config: AutonomousTrainingConfig,
        sentiment_processor: Any,
        start_date: Any = None,
        max_iterations: int = 2,
        llm_context: dict[str, Any] | None = None,
    ) -> AutonomousTrainingReport:
        report = AutonomousTrainingReport(
            ticker=ticker,
            final_config=copy.deepcopy(initial_config.__dict__),
        )

        best_sharpe = -math.inf
        plateau_count = 0

        for iteration in range(1, max_iterations + 1):
            config_snapshot = copy.deepcopy(report.final_config)
            env, _, _ = self.env_builder(
                ticker,
                int(config_snapshot["backtest_days"]),
                db=self.db,
                sentiment_processor=sentiment_processor,
                start_date=start_date,
            )
            if env is None:
                report.stopped_reason = "Unable to build training environment."
                break

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=float(config_snapshot["learning_rate"]),
                ent_coef=float(config_snapshot["entropy_coef"]),
                n_steps=min(512, int(config_snapshot["training_steps"])),
                batch_size=64,
                verbose=0,
            )

            try:
                model.learn(total_timesteps=int(config_snapshot["training_steps"]))
            except Exception as exc:
                logger.error("Autonomous training failed on iteration %s: %s", iteration, exc)
                report.stopped_reason = "Training failed during autonomous tuning."
                break

            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))

            metrics = self._compute_metrics(env.history, 100_000.0)
            recommendation = self.advisor.recommend(
                metrics=metrics,
                history=[s.metrics for s in report.iterations],
                current_config=config_snapshot,
                session_id=None,
                additional_context=llm_context,
            )

            self._persist_iteration(
                ticker=ticker,
                config_snapshot=config_snapshot,
                metrics=metrics,
                recommendation=recommendation,
                iteration=iteration,
            )

            report.iterations.append(
                TrainingIterationSummary(
                    iteration=iteration,
                    metrics=metrics,
                    recommendation=recommendation,
                    config=config_snapshot,
                    completed=True,
                )
            )

            report.best_sharpe = max(report.best_sharpe, metrics.get("sharpe", 0.0))
            improved = metrics.get("sharpe", 0.0) > best_sharpe + float(initial_config.min_sharpe_improvement)
            if improved:
                best_sharpe = metrics.get("sharpe", 0.0)
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= int(initial_config.plateau_iterations):
                report.stopped_on_plateau = True
                report.stopped_reason = (
                    "Terminated early due to plateau in Sharpe improvement."
                )
                break

            if recommendation.config_updates:
                report.final_config = self._apply_changes(report.final_config, recommendation.config_updates)
            else:
                report.stopped_reason = "No further safe recommendation generated."
                break

        return report

    def _persist_iteration(
        self,
        ticker: str,
        config_snapshot: dict[str, Any],
        metrics: dict[str, float],
        recommendation: LLMRecommenderResult,
        iteration: int,
    ) -> None:
        strategy_name = f"{config_snapshot.get('strategy_name', 'PPO_AutoTune')}_iter_{iteration}"
        try:
            self.db.insert_strategy_session(
                ticker=ticker,
                strategy_name=strategy_name,
                training_steps=int(config_snapshot["training_steps"]),
                learning_rate=float(config_snapshot["learning_rate"]),
                entropy_coef=float(config_snapshot["entropy_coef"]),
                backtest_days=int(config_snapshot["backtest_days"]),
                performance_metrics=metrics,
                suggested_change=recommendation.suggested_change,
                notes=recommendation.notes,
            )
        except Exception as exc:
            logger.warning("Failed to persist autonomous session iteration %s: %s", iteration, exc)

    def _apply_changes(
        self,
        base_config: dict[str, Any],
        updates: dict[str, float],
    ) -> dict[str, Any]:
        for key, value in updates.items():
            base_config[key] = value
        return base_config

    @staticmethod
    def _compute_metrics(history: list[dict[str, Any]], initial_balance: float) -> dict[str, float]:
        if len(history) < 2:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
            }
        vals = np.array([row["portfolio_value"] for row in history], dtype=np.float64)
        rets = np.diff(vals) / np.maximum(vals[:-1], 1e-9)
        mean_r = float(rets.mean())
        std_r = float(rets.std())
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 1e-9 else 0.0
        down = rets[rets < 0]
        dstd = float(down.std()) if len(down) > 1 else 1e-9
        sortino = (mean_r / dstd * math.sqrt(252)) if dstd > 1e-9 else 0.0
        running_max = np.maximum.accumulate(vals)
        dd_series = (running_max - vals) / np.maximum(running_max, 1e-9)
        max_dd = float(dd_series.max())
        total_ret = (vals[-1] - initial_balance) / initial_balance
        win_rate = float((rets > 0).mean())
        return {
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "total_return": round(total_ret * 100, 2),
            "win_rate": round(win_rate * 100, 1),
        }
