import sys
import time
import traceback

import requests

sys.path.append("c:/Documents/Projects/stockmarket/newcode")

from main import run_golden_loop
from rl_insider_trader import build_trading_environment


def run_api_health_check():
    url = "http://127.0.0.1:8000/health"
    try:
        resp = requests.get(url, timeout=5)
        print(f"API health check {url}: {resp.status_code} - {resp.json()}")
    except Exception as exc:
        print(f"API health check failed: {exc}")


def run_golden_loop_check():
    try:
        print("Running golden loop for AAPL...")
        signals = run_golden_loop(["AAPL"], days_back=2)
        print(f"Golden loop returned {len(signals)} signals")
        for signal in signals:
            print(signal)
    except Exception:
        traceback.print_exc()


def run_rl_environment_check():
    try:
        print("Building RL training environment for AAPL...")
        env, df, alts = build_trading_environment("AAPL", n_days=30)
        print(f"Environment built: df={len(df)}, alts={len(alts)}")
        print(f"Observation space: {env.observation_space}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Flippy Engine Smoke Test ===")
    run_api_health_check()
    time.sleep(0.5)
    run_golden_loop_check()
    time.sleep(0.5)
    run_rl_environment_check()
    print("=== Smoke test complete ===")
