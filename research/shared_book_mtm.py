"""Research-only minute-close portfolio MTM on actually accepted shared-book fills.

This is *not* executable equity, exchange mark price or liquidation accounting.
A filled position is valued from its archived entry-minute open to the minute
close (and separate same-minute adverse OHLC stress). Historic realized settlement
cash is recognized ONLY at exit + one minute, matching risk_portfolio_audit.
No locked holdout reads. No release authorization.
"""
from __future__ import annotations

import math
import pandas as pd
import numpy as np

from research.execution import net_after_costs, net_after_costs_short
from research.rr_labels import HOLDOUT_START


def audit_shared_book_minute_mtm(
    accepted_trace: list[dict],
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    starting_capital: float = 10000.,
) -> dict:
    if not math.isfinite(starting_capital) or starting_capital <= 0:
        raise ValueError("invalid starting capital")
    if not accepted_trace:
        return {
            "accepted_position_count": 0,
            "observed_minute_count": 0,
            "worst_minute_close_drawdown": 0.,
            "worst_simultaneous_intraminute_proxy_drawdown": 0.,
            "last_settled_equity": float(starting_capital),
            "actual_exchange_liquidation_modeled": False,
            "release_approved": False,
        }

    positions = []
    for raw in accepted_trace:
        symbol = raw["symbol"]
        side = raw["side"]
        if symbol not in bars_by_symbol or side not in ("long", "short"):
            raise ValueError("missing OHLC or invalid side for accepted fill")
        entry, exit_ = pd.Timestamp(raw["entry_at"]), pd.Timestamp(raw["exit_at"])
        if (entry.tzinfo is None or exit_.tzinfo is None
            or entry > exit_ or exit_ >= HOLDOUT_START
            or entry.floor("min") != entry or exit_.floor("min") != exit_):
            raise ValueError("invalid accepted minute/holdout boundary")
        entry, exit_ = entry.tz_convert("UTC"), exit_.tz_convert("UTC")
        notional = float(raw["entry_notional"])
        pnl = float(raw["realized_net_pnl_cash"])
        fee = float(raw["fee_bps_per_side"])
        slip = float(raw["slippage_bps_per_side"])
        if (not all(math.isfinite(v) for v in (notional, pnl, fee, slip))
            or notional <= 0 or fee < 0 or slip < 0
            or fee >= 10000 or slip >= 10000):
            raise ValueError("invalid accepted position cash/costs")
        bars = bars_by_symbol[symbol]
        if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None
            or not bars.index.is_unique or not bars.index.is_monotonic_increasing
            or not {"open", "high", "low", "close", "source_observed"}.issubset(bars.columns)):
            raise ValueError("observed ordered OHLC minutes required")
        window = bars.loc[entry:exit_]
        expected = pd.date_range(entry, exit_, freq="min")
        if not window.index.tz_convert("UTC").as_unit("ns").equals(expected.as_unit("ns")):
            raise ValueError("missing observed accepted-position minute")
        if not window["source_observed"].fillna(False).astype(bool).all():
            raise ValueError("unobserved/synthetic accepted-position minute")
        prices = window[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if (not np.isfinite(prices).all() or np.any(prices <= 0)
            or np.any(prices[:, 1] < prices[:, [0, 2, 3]].max(axis=1))
            or np.any(prices[:, 2] > prices[:, [0, 1, 3]].min(axis=1))):
            raise ValueError("invalid OHLC")
        positions.append((entry, exit_, symbol, side, notional, pnl, fee, slip, window, float(prices[0, 0])))

    # The entry-close and exit-close marks are NOT assumed broker fills.
    first = min(p[0] for p in positions)
    final_exit = max(p[1] for p in positions)
    if final_exit + pd.Timedelta(minutes=1) >= HOLDOUT_START:
        raise ValueError("settlement reaches the locked holdout")
    index = pd.date_range(first, final_exit + pd.Timedelta(minutes=1), freq="min")
    realized_cash = float(starting_capital)
    peak_close = float(starting_capital)
    peak_stress = float(starting_capital)
    worst_close = 0.
    worst_stress = 0.
    # A position still contributes its mark through the exit bar; its actual
    # historical exit cash replaces the proxy mark at exit + 1 minute.
    for minute in index:
        realized_cash += sum(p[5] for p in positions
                             if p[1] + pd.Timedelta(minutes=1) == minute)
        mark_close = 0.
        mark_adverse = 0.
        for entry, exit_, _, side, notional, _, fee, slip, bars, entry_px in positions:
            if not entry <= minute <= exit_:
                continue
            bar = bars.loc[minute]
            if side == "long":
                close_gross = float(bar["close"]) / entry_px - 1
                adverse_gross = float(bar["low"]) / entry_px - 1
                cost_fn = net_after_costs
            else:
                close_gross = 1 - float(bar["close"]) / entry_px
                adverse_gross = 1 - float(bar["high"]) / entry_px
                cost_fn = net_after_costs_short
            mark_close += notional * cost_fn(close_gross, fee, slip)
            mark_adverse += notional * cost_fn(adverse_gross, fee, slip)
        close_equity = realized_cash + mark_close
        adverse_equity = realized_cash + mark_adverse
        if not math.isfinite(close_equity) or not math.isfinite(adverse_equity):
            raise ValueError("invalid equity path")
        peak_close = max(peak_close, close_equity)
        peak_stress = max(peak_stress, close_equity)
        worst_close = max(worst_close, (peak_close - close_equity) / peak_close)
        # Same-minute HIGH/LOW extremes across coins may never have coincided.
        worst_stress = max(worst_stress, (peak_stress - adverse_equity) / peak_stress)

    return {
        "accepted_position_count": len(positions),
        "observed_minute_count": len(index),
        "worst_minute_close_drawdown": float(worst_close),
        "worst_simultaneous_intraminute_proxy_drawdown": float(worst_stress),
        "last_settled_equity": float(realized_cash),
        "entry_and_exit_mark_source": "archived 1m traded OHLC (not exchange mark price)",
        "funding_marked_intraminute": False,
        "full_close_settlement_cash_reconciled": True,
        "actual_exchange_liquidation_modeled": False,
        "release_approved": False,
        "limitations": [
            "Minute-close equity uses archived traded closes and assumed cost functions.",
            "Exit-bar close may differ from modeled stop/target fill; cash switches to exact simulated settlement at exit + 1m.",
            "Intraminute adverse highs/lows across symbols need not have occurred simultaneously; proxy is conservative, not an executable curve.",
            "Financing/funding settlement times, partial fills, impact, real exchange mark prices and cross-margin liquidations not modeled.",
            "The trace contains outcomes only after portfolio admission; never use historical cash/exit for signal ordering.",
        ],
    }
