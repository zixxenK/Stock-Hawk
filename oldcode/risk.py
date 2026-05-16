"""
Risk management and position sizing for manual trade execution.

Based on the 1% risk rule:  never risk more than 1% of your account
on a single trade.  Stop loss is placed at 2× ATR below the entry
(or below the most recent swing low if that's closer).

Usage:
    sizing = calculate_position(
        account_size=50_000,
        entry_price=150.00,
        atr=3.50,
        risk_pct=0.01,
    )
    print(sizing)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSizing:
    ticker: str
    entry_price: float
    stop_loss: float          # Hard stop price
    target_1r: float          # 1:1 R target
    target_2r: float          # 1:2 R target (recommended min for exits)
    target_3r: float          # 1:3 R stretch target
    risk_per_share: float     # $ at risk per share
    shares: int               # Number of shares to buy
    position_value: float     # Total $ to deploy
    dollar_risk: float        # Total $ at risk on the trade
    account_risk_pct: float   # As fraction of account (should be ≤ risk_pct)
    atr_used: float
    notes: list[str]


def calculate_position(
    ticker: str,
    account_size: float,
    entry_price: float,
    atr: float | None,
    risk_pct: float = 0.01,          # 1% default
    atr_stop_multiple: float = 2.0,  # Stop = entry - 2×ATR
    max_position_pct: float = 0.10,  # Max 10% of account in one position
) -> PositionSizing | None:
    """
    Calculate position size, stop loss, and profit targets.

    Args:
        ticker: Ticker symbol (for display).
        account_size: Total account value in dollars.
        entry_price: Intended buy price (use current ask for market orders).
        atr: 14-day ATR.  If None, falls back to 5% of price.
        risk_pct: Fraction of account to risk on this trade.  0.01 = 1%.
        atr_stop_multiple: Place stop this many ATRs below entry.
        max_position_pct: Hard cap on position size as fraction of account.

    Returns:
        PositionSizing dataclass, or None if the trade is not viable.
    """
    if entry_price <= 0:
        return None

    notes: list[str] = []

    # Stop loss placement
    if atr is None or atr <= 0:
        atr_used = entry_price * 0.05      # fallback: 5% of price
        notes.append("ATR unavailable — using 5% price-based stop (wider than ideal)")
    else:
        atr_used = atr

    stop_distance = atr_stop_multiple * atr_used
    stop_loss = entry_price - stop_distance

    if stop_loss <= 0:
        notes.append("Stop loss would be ≤ 0 — consider a smaller ATR multiple")
        stop_loss = entry_price * 0.90     # floor at -10%

    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return None

    # Dollar risk budget
    dollar_risk = account_size * risk_pct

    # Raw share count from risk budget
    shares_raw = dollar_risk / risk_per_share
    shares = max(1, int(shares_raw))

    # Cap by max position size
    max_value = account_size * max_position_pct
    if shares * entry_price > max_value:
        shares = max(1, int(max_value / entry_price))
        notes.append(
            f"Position capped at {max_position_pct:.0%} of account "
            f"(${max_value:,.0f})"
        )

    position_value = shares * entry_price
    actual_dollar_risk = shares * risk_per_share
    actual_risk_pct = actual_dollar_risk / account_size

    # Profit targets (R multiples)
    target_1r = entry_price + risk_per_share          # 1:1
    target_2r = entry_price + 2 * risk_per_share      # 1:2
    target_3r = entry_price + 3 * risk_per_share      # 1:3

    # Quality notes
    if actual_risk_pct > risk_pct * 1.05:
        notes.append(
            f"Actual risk ({actual_risk_pct:.2%}) slightly above target — "
            "round down shares if needed"
        )

    rr_to_target2 = 2.0  # always 1:2 by construction
    if stop_distance / entry_price > 0.15:
        notes.append("Wide stop (>15% of price) — consider waiting for a tighter setup")

    return PositionSizing(
        ticker=ticker,
        entry_price=entry_price,
        stop_loss=round(stop_loss, 2),
        target_1r=round(target_1r, 2),
        target_2r=round(target_2r, 2),
        target_3r=round(target_3r, 2),
        risk_per_share=round(risk_per_share, 2),
        shares=shares,
        position_value=round(position_value, 2),
        dollar_risk=round(actual_dollar_risk, 2),
        account_risk_pct=actual_risk_pct,
        atr_used=round(atr_used, 2),
        notes=notes,
    )


def portfolio_heat(positions: list[PositionSizing], account_size: float) -> dict:
    """
    Summary of total open risk across all planned positions.

    Args:
        positions: List of PositionSizing objects you intend to open.
        account_size: Current account value.

    Returns:
        Dict with total_risk_pct, total_deployed_pct, heat_label.
    """
    total_risk = sum(p.dollar_risk for p in positions)
    total_deployed = sum(p.position_value for p in positions)
    risk_pct = total_risk / account_size if account_size > 0 else 0
    deployed_pct = total_deployed / account_size if account_size > 0 else 0

    if risk_pct < 0.03:
        heat_label = "Cool (room for more positions)"
    elif risk_pct < 0.06:
        heat_label = "Moderate (healthy exposure)"
    elif risk_pct < 0.10:
        heat_label = "Warm (monitor closely)"
    else:
        heat_label = "HOT — consider reducing exposure"

    return {
        "total_dollar_risk": round(total_risk, 2),
        "total_risk_pct": round(risk_pct * 100, 2),
        "total_deployed": round(total_deployed, 2),
        "total_deployed_pct": round(deployed_pct * 100, 2),
        "heat_label": heat_label,
        "num_positions": len(positions),
    }
