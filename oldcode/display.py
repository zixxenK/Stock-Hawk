"""
Terminal display and CSV export for scan results.

Uses the `rich` library for a formatted, colour-coded leaderboard.
Falls back to plain-text if rich is not installed.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

from .risk import PositionSizing, calculate_position
from .scoring import MomentumScore

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

GRADE_COLORS = {
    "A+": "bold bright_green",
    "A":  "green",
    "B+": "bright_yellow",
    "B":  "yellow",
    "C":  "white",
    "D":  "dim",
    "F":  "red dim",
}


def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def _fmt_float(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_bool(v: bool | None) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"


def print_leaderboard(results: list[MomentumScore], top_n: int = 30) -> None:
    """Print the ranked momentum leaderboard to the terminal."""
    shown = results[:top_n]

    if not HAS_RICH:
        _print_plain(shown)
        return

    console = Console()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    console.print()
    console.print(
        Panel(
            f"[bold white]Momentum Scanner[/bold white]  ·  [dim]{now}[/dim]",
            subtitle="[dim]Long-term momentum · manual execution[/dim]",
            border_style="gold1",
        )
    )

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        border_style="dim",
        show_lines=False,
        pad_edge=True,
    )

    table.add_column("Rank", justify="right", style="dim", width=5)
    table.add_column("Ticker", justify="left", width=7)
    table.add_column("Grade", justify="center", width=6)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Price", justify="right", width=9)
    table.add_column("12-1m Mom", justify="right", width=10)
    table.add_column("vs SPY", justify="right", width=9)
    table.add_column("52wk High", justify="right", width=10)
    table.add_column("RSI", justify="right", width=5)
    table.add_column("ADX", justify="right", width=5)
    table.add_column("GX", justify="center", width=4, header="GX")  # Golden cross
    table.add_column("Vol×", justify="right", width=5)
    table.add_column("Signals", justify="left")

    for s in shown:
        grade_style = GRADE_COLORS.get(s.grade, "white")

        gx_str = "✓" if s.golden_cross else ("✗" if s.golden_cross is False else "—")
        gx_color = "green" if s.golden_cross else ("red" if s.golden_cross is False else "dim")

        signal_preview = " · ".join(s.signals[:2]) if s.signals else ""
        if s.warnings:
            signal_preview += (
                (" · " if signal_preview else "")
                + f"[dim red]{s.warnings[0]}[/dim red]"
            )

        score_style = "bold bright_green" if s.score >= 80 else (
            "bright_yellow" if s.score >= 60 else "dim"
        )

        table.add_row(
            f"#{s.rank}",
            f"[bold]{s.ticker}[/bold]",
            f"[{grade_style}]{s.grade}[/{grade_style}]",
            f"[{score_style}]{s.score:.1f}[/{score_style}]",
            f"${s.price:,.2f}",
            _fmt_pct(s.mom_12_1),
            _fmt_pct(s.rs_vs_spy),
            _fmt_pct(s.high_52w_pct and (s.high_52w_pct - 1)),
            _fmt_float(s.rsi, 0),
            _fmt_float(s.adx, 0),
            f"[{gx_color}]{gx_str}[/{gx_color}]",
            _fmt_float(s.vol_trend, 1),
            signal_preview,
        )

    console.print(table)
    console.print(
        f"[dim]GX = Golden Cross (50 SMA > 200 SMA)  ·  Vol× = volume vs 60-day avg  ·  "
        f"Score = 0-100 composite momentum score[/dim]\n"
    )


def print_detail(s: MomentumScore, sizing: PositionSizing | None = None) -> None:
    """Print a detailed card for a single ticker."""
    if not HAS_RICH:
        _print_plain_detail(s, sizing)
        return

    console = Console()
    grade_style = GRADE_COLORS.get(s.grade, "white")

    lines = [
        f"[bold]Price[/bold]  ${s.price:,.2f}   "
        f"[bold]Score[/bold]  {s.score:.1f}/100   "
        f"[bold]Grade[/bold]  [{grade_style}]{s.grade}[/{grade_style}]",
        "",
        "[bold]Momentum[/bold]",
        f"  12-1m: {_fmt_pct(s.mom_12_1)}    "
        f"6m: {_fmt_pct(s.mom_6m)}    "
        f"3m: {_fmt_pct(s.mom_3m)}",
        f"  vs SPY (12m): {_fmt_pct(s.rs_vs_spy)}    "
        f"RS Ratio: {_fmt_float(s.rs_ratio, 2)}",
        f"  Sharpe Mom: {_fmt_float(s.sharpe_mom, 2)}",
        "",
        "[bold]Trend[/bold]",
        f"  Golden Cross: {_fmt_bool(s.golden_cross)}    "
        f"Price vs 200 SMA: {_fmt_pct(s.price_vs_200)}",
        f"  ADX: {_fmt_float(s.adx, 1)}    RSI: {_fmt_float(s.rsi, 1)}    "
        f"MACD Bull: {_fmt_bool(s.macd_bullish)}",
        "",
        "[bold]Levels & Supply/Demand[/bold]",
        f"  52-week high proximity: {_fmt_pct(s.high_52w_pct and (s.high_52w_pct - 1))}",
        f"  Volume trend: {_fmt_float(s.vol_trend, 2)}×    "
        f"Hist Vol: {_fmt_pct(s.hist_vol)}",
    ]

    if s.signals:
        lines += ["", "[bold green]Signals ✓[/bold green]"]
        for sig in s.signals:
            lines.append(f"  [green]+ {sig}[/green]")

    if s.warnings:
        lines += ["", "[bold red]Warnings ⚠[/bold red]"]
        for w in s.warnings:
            lines.append(f"  [red]! {w}[/red]")

    if sizing:
        lines += [
            "",
            "[bold]Position Sizing (1% account risk)[/bold]",
            f"  Entry:    ${sizing.entry_price:,.2f}",
            f"  Stop:     ${sizing.stop_loss:,.2f}  "
            f"({(sizing.entry_price - sizing.stop_loss) / sizing.entry_price:.1%} below entry)",
            f"  Target 1R: ${sizing.target_1r:,.2f}",
            f"  Target 2R: ${sizing.target_2r:,.2f}  ← recommended minimum exit",
            f"  Target 3R: ${sizing.target_3r:,.2f}",
            f"  Shares:   {sizing.shares:,}  "
            f"(${sizing.position_value:,.0f} deployed)",
            f"  $ at risk: ${sizing.dollar_risk:,.0f}  "
            f"({sizing.account_risk_pct:.2%} of account)",
        ]
        for note in sizing.notes:
            lines.append(f"  [dim yellow]Note: {note}[/dim yellow]")

    factor_lines = "  ".join(
        f"{k}: {v:.2f}" for k, v in sorted(s.factor_scores.items())
    )
    lines += ["", f"[dim]Factor scores: {factor_lines}[/dim]"]

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold]{s.ticker}[/bold]  —  Rank #{s.rank}",
            border_style=grade_style.split()[-1] if grade_style else "white",
        )
    )


def export_csv(results: list[MomentumScore], path: str | Path | None = None) -> Path:
    """
    Export scan results to a CSV file.

    Args:
        results: List of MomentumScore objects.
        path: Output path.  Defaults to ./reports/scan_YYYYMMDD_HHMM.csv

    Returns:
        Path to the written file.
    """
    if path is None:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        path = reports_dir / f"scan_{ts}.csv"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "rank", "ticker", "grade", "score", "price",
        "mom_12_1", "mom_6m", "mom_3m",
        "rs_vs_spy", "rs_ratio",
        "high_52w_pct", "sharpe_mom",
        "golden_cross", "price_vs_200",
        "adx", "vol_trend", "rsi", "macd_bullish",
        "atr", "hist_vol",
        "signals", "warnings",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in results:
            writer.writerow({
                "rank": s.rank,
                "ticker": s.ticker,
                "grade": s.grade,
                "score": round(s.score, 2),
                "price": s.price,
                "mom_12_1": _fmt_pct(s.mom_12_1) if s.mom_12_1 is not None else "",
                "mom_6m": _fmt_pct(s.mom_6m) if s.mom_6m is not None else "",
                "mom_3m": _fmt_pct(s.mom_3m) if s.mom_3m is not None else "",
                "rs_vs_spy": _fmt_pct(s.rs_vs_spy) if s.rs_vs_spy is not None else "",
                "rs_ratio": round(s.rs_ratio, 3) if s.rs_ratio is not None else "",
                "high_52w_pct": round(s.high_52w_pct, 4) if s.high_52w_pct is not None else "",
                "sharpe_mom": round(s.sharpe_mom, 2) if s.sharpe_mom is not None else "",
                "golden_cross": s.golden_cross if s.golden_cross is not None else "",
                "price_vs_200": _fmt_pct(s.price_vs_200) if s.price_vs_200 is not None else "",
                "adx": round(s.adx, 1) if s.adx is not None else "",
                "vol_trend": round(s.vol_trend, 2) if s.vol_trend is not None else "",
                "rsi": round(s.rsi, 1) if s.rsi is not None else "",
                "macd_bullish": s.macd_bullish if s.macd_bullish is not None else "",
                "atr": round(s.atr, 2) if s.atr is not None else "",
                "hist_vol": _fmt_pct(s.hist_vol) if s.hist_vol is not None else "",
                "signals": "; ".join(s.signals),
                "warnings": "; ".join(s.warnings),
            })

    return path


def _print_plain(results: list[MomentumScore]) -> None:
    """Fallback plain-text leaderboard (no rich dependency)."""
    print(f"\n{'MOMENTUM LEADERBOARD':=^80}")
    header = f"{'RK':>3}  {'TICKER':<7} {'GR':>3} {'SCORE':>6} {'PRICE':>9} "
    header += f"{'12-1M':>8} {'vsSPY':>8} {'RSI':>5} {'ADX':>5}"
    print(header)
    print("-" * 80)
    for s in results:
        print(
            f"#{s.rank:<2}  {s.ticker:<7} {s.grade:>3} {s.score:>6.1f} "
            f"${s.price:>8,.2f} "
            f"{_fmt_pct(s.mom_12_1):>8} {_fmt_pct(s.rs_vs_spy):>8} "
            f"{_fmt_float(s.rsi, 0):>5} {_fmt_float(s.adx, 0):>5}"
        )
    print()


def _print_plain_detail(s: MomentumScore, sizing: PositionSizing | None) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {s.ticker}  |  Score: {s.score:.1f}/100  |  Grade: {s.grade}")
    print(f"  Price: ${s.price:,.2f}")
    print(f"  12-1m Mom: {_fmt_pct(s.mom_12_1)}  |  vs SPY: {_fmt_pct(s.rs_vs_spy)}")
    print(f"  Golden Cross: {_fmt_bool(s.golden_cross)}  |  ADX: {_fmt_float(s.adx, 1)}")
    if s.signals:
        print("  Signals: " + "; ".join(s.signals))
    if s.warnings:
        print("  Warnings: " + "; ".join(s.warnings))
    if sizing:
        print(f"\n  SIZING:")
        print(f"    Entry: ${sizing.entry_price:,.2f}   Stop: ${sizing.stop_loss:,.2f}")
        print(f"    1R: ${sizing.target_1r:,.2f}   2R: ${sizing.target_2r:,.2f}")
        print(f"    Shares: {sizing.shares:,}   $ Risk: ${sizing.dollar_risk:,.0f}")
    print("=" * 60)
