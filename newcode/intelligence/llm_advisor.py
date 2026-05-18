from __future__ import annotations

import math
import os
import requests
from dataclasses import dataclass, field
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRANSFORMERS_AVAILABLE = False


@dataclass
class LLMRecommenderResult:
    suggested_change: str
    notes: str
    config_updates: dict[str, float] = field(default_factory=dict)
    advisor: str = "local_rule_fallback"
    confidence: float = 0.75


class LocalLLMAdvisor:
    """Local advisor that generates safe, auditable tuning recommendations.

    This class is intentionally designed to support a real local model in the
    future via a configured transformer path, but it currently provides a
    deterministic rule-based fallback so the pipeline remains functional.
    """

    DEFAULT_LEARNING_RATE_RANGE = (1e-6, 1e-3)
    DEFAULT_ENTROPY_RANGE = (0.0, 0.15)
    DEFAULT_TRAINING_STEPS_RANGE = (50_000, 5_000_000)
    DEFAULT_BACKTEST_DAYS_RANGE = (400, 5200)

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("LLM_ADVISOR_MODEL_PATH")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL_NAME", "")
        self.use_ollama = bool(self.ollama_model)
        self.use_model = bool((self.model_path and TRANSFORMERS_AVAILABLE) or self.use_ollama)

    def recommend(
        self,
        metrics: dict[str, float],
        history: list[dict[str, Any]],
        current_config: dict[str, float],
        session_id: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> LLMRecommenderResult:
        if self.use_ollama:
            return self._recommend_from_ollama(metrics, history, current_config, session_id, additional_context)
        if self.model_path and TRANSFORMERS_AVAILABLE:
            return self._recommend_from_local_model(metrics, history, current_config, session_id, additional_context)
        return self._rule_based_recommendation(metrics, history, current_config, additional_context)

    def _recommend_from_local_model(
        self,
        metrics: dict[str, float],
        history: list[dict[str, Any]],
        current_config: dict[str, float],
        session_id: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> LLMRecommenderResult:
        prompt = self._build_prompt(metrics, history, current_config, session_id, additional_context)
        text = self._call_transformer(prompt)
        return self._parse_transformer_output(text, current_config)

    def _recommend_from_ollama(
        self,
        metrics: dict[str, float],
        history: list[dict[str, Any]],
        current_config: dict[str, float],
        session_id: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> LLMRecommenderResult:
        prompt = self._build_prompt(metrics, history, current_config, session_id, additional_context)
        text = self._call_ollama(prompt)
        return self._parse_transformer_output(text, current_config)

    def _build_prompt(
        self,
        metrics: dict[str, float],
        history: list[dict[str, Any]],
        current_config: dict[str, float],
        session_id: str | None,
        additional_context: dict[str, Any] | None = None,
    ) -> str:
        history_text = "\n".join(
            f"- Iteration {idx+1}: {h.get('sharpe', 0.0):.3f} Sharpe, "
            f"{h.get('max_drawdown', 0.0):.1f}% drawdown, {h.get('win_rate', 0.0):.1f}% win rate"
            for idx, h in enumerate(history[-5:])
        ) or "No previous tuning history."

        context_text = ""
        if additional_context:
            safe_context = json.dumps(additional_context, default=str)
            context_text = f"Additional context: {safe_context}. "

        return (
            "You are a local model advising on PPO training for a trading agent. "
            "Use only the numerical metrics and safe hyperparameter ranges below. "
            "Return valid JSON with exactly these fields: "
            "config_updates, suggested_change, notes, confidence. "
            "config_updates may include learning_rate, entropy_coef, training_steps, and backtest_days only. "
            "Do not change any other configuration. "
            "Only recommend values within the current operational bounds. "
            f"Current metrics: {metrics}. "
            f"History summary: {history_text}. "
            f"Current config: {current_config}. "
            f"Session ID: {session_id or 'none'}. "
            f"{context_text}"
            "If no safe change is required, return config_updates as an empty object "
            "and a suggested_change that states no change is recommended."
        )

    def _call_transformer(self, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = generator(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"]

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_url.rstrip('/')}/api/models/{self.ollama_model}/outputs"
        payload = {
            "input": prompt,
            "options": {
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and isinstance(choices, list):
                    message = choices[0].get("message") or {}
                    text = message.get("content") or choices[0].get("content") or ""
                    return str(text)
            return ""
        except Exception as exc:
            return ""

    def _parse_transformer_output(
        self,
        text: str,
        current_config: dict[str, float],
    ) -> LLMRecommenderResult:
        try:
            payload = self._extract_json(text)
            updates = payload.get("config_updates")
            if not isinstance(updates, dict):
                updates = {
                    k: v
                    for k, v in payload.items()
                    if k in {
                        "learning_rate",
                        "entropy_coef",
                        "training_steps",
                        "backtest_days",
                    }
                }
            validated = self._sanitize_updates(updates or {}, current_config)
            return LLMRecommenderResult(
                suggested_change=str(payload.get("suggested_change", "")) or "No change recommended.",
                notes=str(payload.get("notes", "")) or "Model returned a safe recommendation.",
                config_updates=validated,
                advisor="local_ollama_model" if self.use_ollama else "local_transformer_model",
                confidence=float(payload.get("confidence", 0.75)),
            )
        except Exception:
            return self._rule_based_recommendation(metrics={}, history=[], current_config=current_config, additional_context=None)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Unable to parse JSON from transformer output")
        return eval(text[start : end + 1])  # noqa: S307 -- controlled parse of generated model text

    def _rule_based_recommendation(
        self,
        metrics: dict[str, float],
        history: list[dict[str, Any]],
        current_config: dict[str, float],
        additional_context: dict[str, Any] | None = None,
    ) -> LLMRecommenderResult:
        metrics = {k: float(metrics.get(k, 0.0)) for k in [
            "total_return", "sharpe", "sortino", "max_drawdown", "win_rate",
        ]}
        updates: dict[str, float] = {}
        message = "Maintain the current training setup and collect more data."

        if metrics["total_return"] < -5.0:
            updates["learning_rate"] = self._clip(
                current_config.get("learning_rate", 3e-4) / 2.0,
                self.DEFAULT_LEARNING_RATE_RANGE,
            )
            updates["entropy_coef"] = self._clip(
                current_config.get("entropy_coef", 0.02) + 0.01,
                self.DEFAULT_ENTROPY_RANGE,
            )
            updates["training_steps"] = self._clip(
                current_config.get("training_steps", 250_000) * 1.2,
                self.DEFAULT_TRAINING_STEPS_RANGE,
            )
            message = (
                "Strong negative performance detected. Reduce the learning rate, increase exploration, "
                "and allocate a slightly larger training budget for stability."
            )
        elif metrics["sharpe"] < 0.75:
            updates["learning_rate"] = self._clip(
                current_config.get("learning_rate", 3e-4) * 0.75,
                self.DEFAULT_LEARNING_RATE_RANGE,
            )
            updates["entropy_coef"] = self._clip(
                current_config.get("entropy_coef", 0.02) + 0.005,
                self.DEFAULT_ENTROPY_RANGE,
            )
            message = (
                "Low Sharpe ratio. Soften policy updates and improve exploration to prevent overfitting."
            )
        elif metrics["max_drawdown"] > 15.0:
            updates["backtest_days"] = self._clip(
                current_config.get("backtest_days", 2520) - 250,
                self.DEFAULT_BACKTEST_DAYS_RANGE,
            )
            updates["entropy_coef"] = self._clip(
                current_config.get("entropy_coef", 0.02) + 0.01,
                self.DEFAULT_ENTROPY_RANGE,
            )
            message = (
                "High drawdown suggests regime mismatch. Shorten the historical window and increase exploration."
            )
        elif metrics["win_rate"] < 45.0:
            updates["training_steps"] = self._clip(
                current_config.get("training_steps", 250_000) * 1.25,
                self.DEFAULT_TRAINING_STEPS_RANGE,
            )
            message = (
                "Low win rate. Extend training budget to let the agent learn the signal distribution more fully."
            )
        elif metrics["sharpe"] >= 1.0 and metrics["max_drawdown"] < 12.0:
            updates["training_steps"] = self._clip(
                current_config.get("training_steps", 250_000) * 1.5,
                self.DEFAULT_TRAINING_STEPS_RANGE,
            )
            updates["backtest_days"] = self._clip(
                current_config.get("backtest_days", 2520) + 250,
                self.DEFAULT_BACKTEST_DAYS_RANGE,
            )
            message = (
                "Stable performance detected. Increase training budget and include a larger sample of history."
            )

        validated = self._sanitize_updates(updates, current_config)
        notes = (
            f"Current metrics: Sharpe={metrics['sharpe']:.3f}, Return={metrics['total_return']:+.1f}%, "
            f"Drawdown={metrics['max_drawdown']:.1f}%, WinRate={metrics['win_rate']:.1f}%. "
            "The advisor suggests a safe configuration adjustment."
        )
        if not validated:
            validated = {}
            message = "No safe tuning change is recommended at this time."
            notes = "Performance is within expected bounds; continue collecting more training data."

        return LLMRecommenderResult(
            suggested_change=message,
            notes=notes,
            config_updates=validated,
            advisor="local_rule_fallback",
            confidence=0.85,
        )

    def _sanitize_updates(
        self,
        candidate_updates: dict[str, float],
        current_config: dict[str, float],
    ) -> dict[str, float]:
        validated: dict[str, float] = {}
        for key, value in candidate_updates.items():
            if key == "learning_rate":
                validated[key] = self._clip(value, self.DEFAULT_LEARNING_RATE_RANGE)
            elif key == "entropy_coef":
                validated[key] = self._clip(value, self.DEFAULT_ENTROPY_RANGE)
            elif key == "training_steps":
                validated[key] = int(self._clip(value, self.DEFAULT_TRAINING_STEPS_RANGE))
            elif key == "backtest_days":
                validated[key] = int(self._clip(value, self.DEFAULT_BACKTEST_DAYS_RANGE))
        return validated

    @staticmethod
    def _clip(value: float, bounds: tuple[float, float]) -> float:
        return float(max(bounds[0], min(bounds[1], value)))
