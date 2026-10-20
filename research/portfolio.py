"""Research-only capital-constrained Fishnet portfolio accounting.

Inputs are historical simulated fills whose net_return ALREADY includes the
specified assumed fee/slippage round trip. Do not deduct costs twice.
No leverage, borrowing, live order placement, or automatic model promotion.
Equity and drawdown are measured only at realized closes (not intrabar MTM).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


NET_30D_TARGET = 0.07


@dataclass(frozen=True)
class CandidateFill:
    symbol: str
    entry_at: str
    exit_at: str
    net_return: float


def portfolio_audit(
    fills: list[CandidateFill],
    *,
    starting_capital: float = 10000.0,
    stake_fraction: float = 0.10,
    max_concurrent_positions: int = 2,
) -> dict:
    """Allocate a fixed fraction of contemporaneous equity per trade.

    At every entry, settle positions with exits <= that entry. Reserve notional
    across overlapping positions and forbid overlapping positions in the same
    symbol. No leverage; idle cash is included in the equity denominator.
    Exit timestamps identify 1-minute bars, so capital remains reserved through
    that exit minute. Positions held through a price excursion are *not* MTM valued here:
    max_realized_drawdown understates true mark-to-market drawdown.
    """
    if not math.isfinite(starting_capital) or starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if not math.isfinite(stake_fraction) or not 0 < stake_fraction <= 1:
        raise ValueError("stake_fraction must be within (0, 1]")
    if type(max_concurrent_positions) is not int or max_concurrent_positions < 1:
        raise ValueError("max_concurrent_positions must be a positive integer")

    ordered = []
    for fill in fills:
        start, end = pd.Timestamp(fill.entry_at), pd.Timestamp(fill.exit_at)
        if (start.tzinfo is None or end.tzinfo is None or start > end
            or not isinstance(fill.symbol, str) or not fill.symbol.strip()
            or not math.isfinite(fill.net_return) or fill.net_return <= -1):
            raise ValueError("invalid or timezone-naive fill")
        ordered.append((start.tz_convert("UTC"), end.tz_convert("UTC"), fill))
    # Do not tie-break identical entry times using future exit timestamps.
    # Stable ordering preserves the caller's declared signal precedence.
    ordered.sort(key=lambda r: (r[0], r[2].symbol))

    equity = float(starting_capital)
    peak = equity
    max_realized_drawdown = 0.0
    positions: list[tuple[pd.Timestamp, str, float, float]] = []
    executed = 0
    skipped = {"capacity": 0, "symbol_overlap": 0, "capital": 0}

    def settle(up_to: pd.Timestamp | None) -> None:
        nonlocal equity, peak, max_realized_drawdown
        due = [p for p in positions if up_to is None or p[0] <= up_to]
        due.sort(key=lambda p: (p[0], p[1]))
        for p in due:
            equity += p[2] * p[3]
            peak = max(peak, equity)
            max_realized_drawdown = max(max_realized_drawdown, (peak - equity) / peak)
            positions.remove(p)

    for start, end, fill in ordered:
        settle(start)
        if any(p[1] == fill.symbol for p in positions):
            skipped["symbol_overlap"] += 1
            continue
        if len(positions) >= max_concurrent_positions:
            skipped["capacity"] += 1
            continue
        reserved = sum(p[2] for p in positions)
        available = max(0.0, equity - reserved)
        notional = min(equity * stake_fraction, available)
        if notional <= 0 or not math.isfinite(notional):
            skipped["capital"] += 1
            continue
        positions.append((end + pd.Timedelta(minutes=1), fill.symbol, notional, fill.net_return))
        executed += 1
    settle(None)

    return {
        "starting_capital": float(starting_capital),
        "final_equity": float(equity),
        "net_equity_return": float(equity / starting_capital - 1),
        "target_net_30d_return": NET_30D_TARGET,
        "net_30d_goal_met_in_development": bool(equity / starting_capital - 1 >= NET_30D_TARGET),
        "candidate_fills": len(fills),
        "executed_fills": executed,
        "skipped_fills": skipped,
        "max_realized_drawdown": float(max_realized_drawdown),
        "stake_fraction_of_equity": stake_fraction,
        "max_concurrent_positions": max_concurrent_positions,
        "leverage": 1.0,
        "costs": "net_return assumed already includes two-sided fees and slippage; never deducted twice",
        "limitations": [
            "Evaluated over a 30-day development window, NOT a guarantee of calendar-month return.",
            "Uses assumed delayed fills, not executable market orders.",
            "No funding/borrow costs, partial fills, portfolio mark-to-market drawdown or market impact.",
            "Signals in prior tests were retrospectively selected from a historical dataset.",
            "Results are development-only and do not authorize deployment or live orders.",
        ],
        "release_approved": False,
    }
