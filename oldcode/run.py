"""
Momentum Scanner — Clean Entry Point
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import warnings
from typing import List, Optional

# 1. Silencing the yfinance noise
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# 2. Setup Pathing
# Ensures the local 'momentum_scanner' package is found
root_path = Path(__file__).parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# 3. Clean Imports (Removed the broken __import__ logic)
try:
    from momentum_scanner import (
        calculate_position,
        export_csv,
        get_universe,       
        portfolio_heat,
        print_detail,
        print_leaderboard,
        run_scan,
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not find 'momentum_scanner' components.")
    print(f"Details: {e}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with type safety."""
    p = argparse.ArgumentParser(description="Long-term momentum stock scanner")
    
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    group.add_argument("--nasdaq-only", action="store_true", help="Scan Nasdaq 100")

    p.add_argument("--account", type=float, help="Account size for sizing")
    p.add_argument("--risk", type=float, default=0.01, help="Risk % (0.01 = 1%)")
    p.add_argument("--csv", action="store_true", help="Export to CSV")
    p.add_argument("--top", type=int, default=30, help="Results to display")
    p.add_argument("--detail", type=int, help="Show detail cards for top N")

    return p.parse_args()


def main() -> None:
    """Execution logic."""
    args = parse_args()
    
    # Get universe based on flags
    tickers = get_universe(nasdaq_only=args.nasdaq_only) if not args.tickers else args.tickers

    if not tickers:
        print("ERROR: No tickers loaded.")
        return

    # Run the scan
    results = run_scan(tickers)

    if not results:
        print("No stocks met the momentum criteria.")
        return

    # Visualization
    print_leaderboard(results, top=args.top)

    if args.detail:
        print_detail(results, count=args.detail)

    if args.csv:
        export_csv(results)

    if args.account:
        calculate_position(results, args.account, args.risk)
        portfolio_heat(results)

if __name__ == "__main__":
    main()