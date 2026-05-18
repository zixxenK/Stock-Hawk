from intelligence.llm_advisor import LocalLLMAdvisor


def test_local_llm_advisor_fallback_returns_safe_update():
    advisor = LocalLLMAdvisor()
    current_config = {
        "training_steps": 500_000,
        "learning_rate": 0.0003,
        "entropy_coef": 0.02,
        "backtest_days": 2520,
    }
    metrics = {
        "total_return": -8.2,
        "sharpe": 0.12,
        "sortino": 0.20,
        "max_drawdown": 18.5,
        "win_rate": 38.0,
    }

    recommendation = advisor.recommend(
        metrics,
        history=[],
        current_config=current_config,
        additional_context={
            "focus_settings": {
                "insider_flow_weight": 1.2,
                "congress_flow_weight": 1.0,
            },
            "watchlist_tickers": ["AAPL", "MSFT"],
        },
    )

    assert recommendation.suggested_change
    assert recommendation.notes
    assert recommendation.config_updates["learning_rate"] <= current_config["learning_rate"]
    assert recommendation.config_updates["entropy_coef"] >= current_config["entropy_coef"]
    assert recommendation.config_updates["training_steps"] >= current_config["training_steps"]
