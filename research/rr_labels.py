"""Fishnet experimental cost-net 3.2R LONG/SHORT event labels (historical only).

Decision features are known through minute t. Entry is an ASSUMED fill at
the open of minute t+2, never at the old t+1 label entry. Bracket exits use
1-minute OHLC with STOP priority on ambiguous candles, adverse gap opens,
and a fixed max holding horizon. Net reward is defined against the FULL
cost-inclusive loss at the planned stop: +3.2R is not +3.2% of notional.

These labels are research targets, not a prediction, execution or profit claim.
Do not evaluate labels intersecting the locked 2026-08-17+ holdout.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from research.execution import net_after_costs, net_after_costs_short

HOLDOUT_START = pd.Timestamp("2026-08-17T00:00:00Z")
REWARD_MULTIPLE = 3.2


def rr_price_target(
    stop_fraction: float, side: str, *,
    reward_multiple: float = REWARD_MULTIPLE,
    fee_bps_per_side: float = 6.,
    slippage_bps_per_side: float = 3.,
) -> tuple[float, float]:
    """Return (gross favorable price move, net loss fraction at planned stop).

    For short, the exact side-aware entry sell proceeds and buy-back costs
    are used. Actual gap/partial/liquidation losses can exceed 1R.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if (not math.isfinite(stop_fraction) or not 0 < stop_fraction < 1
        or not math.isfinite(reward_multiple) or reward_multiple <= 0):
        raise ValueError("invalid stop or reward multiple")
    values = (fee_bps_per_side, slippage_bps_per_side)
    if not all(math.isfinite(x) and 0 <= x < 10000 for x in values):
        raise ValueError("invalid execution costs")
    fee, slip = fee_bps_per_side / 10000, slippage_bps_per_side / 10000
    retention = (1 - fee) * (1 - slip) / ((1 + fee) * (1 + slip))
    if side == "long":
        stop_loss = -net_after_costs(-stop_fraction, fee_bps_per_side, slippage_bps_per_side)
        target_gross = (1 + reward_multiple * stop_loss) / retention - 1
    else:
        stop_loss = -net_after_costs_short(-stop_fraction, fee_bps_per_side, slippage_bps_per_side)
        target_gross = 1 - (1 - reward_multiple * stop_loss) * retention
    if not math.isfinite(target_gross) or not 0 < target_gross < 1:
        raise ValueError("cost-net target is impossible within supported price range")
    return float(target_gross), float(stop_loss)


def label_rr_events(
    bars: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
    stop_fractions: np.ndarray,
    *,
    side: str,
    horizon_minutes: int = 60,
    reward_multiple: float = REWARD_MULTIPLE,
    entry_delay_minutes: int = 2,
    fee_bps_per_side: float = 6.,
    slippage_bps_per_side: float = 3.,
    cutoff_exclusive: pd.Timestamp = HOLDOUT_START,
) -> pd.DataFrame:
    """Label historical risk/reward outcomes for the supplied causal decisions.

    An event is invalid/NaN when any minute t through the full exit horizon
    is unobserved, outside the accessible pre-holdout data, or has invalid
    OHLC. Timeout counts as NOT hitting 3.2R but retains its realized net R.
    Outcome labels never enter training features.
    """
    if side not in ("long", "short"):
        raise ValueError("invalid side")
    if (not isinstance(horizon_minutes, int) or horizon_minutes < 1 or
        not isinstance(entry_delay_minutes, int) or entry_delay_minutes < 2):
        raise ValueError("invalid horizon or entry delay")
    cutoff = pd.Timestamp(cutoff_exclusive)
    if cutoff.tzinfo is None or cutoff > HOLDOUT_START:
        raise ValueError("cannot access the locked holdout")
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None
        or not bars.index.is_unique or not bars.index.is_monotonic_increasing):
        raise ValueError("historical bars must have a unique, ordered UTC-aware minute index")
    if not {"open", "high", "low", "close", "source_observed"}.issubset(bars.columns):
        raise ValueError("observed OHLC bars required")
    decisions = pd.DatetimeIndex(decision_times)
    if decisions.tz is None or decisions.has_duplicates:
        raise ValueError("decision times must be timezone-aware and unique")
    stops = np.asarray(stop_fractions, dtype=np.float64)
    if stops.shape != (len(decisions),):
        raise ValueError("each decision requires a planned stop distance")
    if np.any(~np.isfinite(stops)) or np.any((stops <= 0) | (stops >= 1)):
        raise ValueError("stop fractions must be finite in (0,1)")
    # Reject physically impossible target geometry up front; this is not an
    # invitation to shrink expected transaction costs or alter labels.
    price_moves = np.empty(len(stops), dtype=float)
    stop_losses = np.empty(len(stops), dtype=float)
    for k, stop in enumerate(stops):
        price_moves[k], stop_losses[k] = rr_price_target(
            float(stop), side, reward_multiple=reward_multiple,
            fee_bps_per_side=fee_bps_per_side, slippage_bps_per_side=slippage_bps_per_side,
        )

    index = bars.index
    decision_idx = index.get_indexer(decisions)
    first = decision_idx + entry_delay_minutes
    last = first + horizon_minutes - 1
    n = len(bars)
    valid = (decision_idx >= 0) & (first < n) & (last < n) & (decisions < cutoff)
    in_range = np.flatnonzero(valid)
    if len(in_range):
        valid[in_range] &= (index[last[in_range]] < cutoff).to_numpy() if hasattr(
            index[last[in_range]] < cutoff, "to_numpy") else index[last[in_range]] < cutoff

    source = bars["source_observed"].fillna(False).to_numpy(dtype=bool)
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    consistent = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) &
                  (prices[:, 1] >= prices[:, [0, 2, 3]].max(axis=1)) &
                  (prices[:, 2] <= prices[:, [0, 1, 3]].min(axis=1)))
    # Source continuity includes the decision bar and delayed-entry bars.
    good = source & consistent
    # Timedelta comparisons are resolution-independent (pandas may store us/ns).
    sequential = np.r_[True, (index[1:] - index[:-1]) == pd.Timedelta(minutes=1)]
    good &= sequential
    prefix = np.r_[0, np.cumsum((~good).astype(np.int64))]
    good_idx = np.flatnonzero(valid)
    if len(good_idx):
        valid[good_idx] = (prefix[last[good_idx] + 1] - prefix[decision_idx[good_idx]]) == 0

    outcomes = np.full(len(stops), np.nan, dtype=float)
    realized_r = np.full(len(stops), np.nan, dtype=float)
    gross_returns = np.full(len(stops), np.nan, dtype=float)
    reasons = np.full(len(stops), "invalid", dtype=object)
    entry_at = np.full(len(stops), pd.NaT, dtype=object)
    exit_at = np.full(len(stops), pd.NaT, dtype=object)
    target_gross = price_moves.copy()

    chosen = np.flatnonzero(valid)
    if len(chosen):
        ent = prices[first[chosen], 0]
        st = stops[chosen]
        moves = price_moves[chosen]
        stop_px = ent * (1 - st if side == "long" else 1 + st)
        target_px = ent * (1 + moves if side == "long" else 1 - moves)
        chosen_exit = last[chosen].copy()
        exit_price = prices[chosen_exit, 3].copy()
        exit_reasons = np.full(len(chosen), "time_limit", dtype=object)
        pending = np.ones(len(chosen), dtype=bool)
        for offset in range(horizon_minutes):
            active = np.flatnonzero(pending)
            if not len(active):
                break
            bar_index = first[chosen[active]] + offset
            opens, highs, lows = prices[bar_index, 0], prices[bar_index, 1], prices[bar_index, 2]
            if side == "long":
                gap_stop = opens <= stop_px[active]
                stop = ~gap_stop & (lows <= stop_px[active])
                target = ~gap_stop & ~stop & (highs >= target_px[active])
            else:
                gap_stop = opens >= stop_px[active]
                stop = ~gap_stop & (highs >= stop_px[active])
                target = ~gap_stop & ~stop & (lows <= target_px[active])
            for hit, label, fill in (
                (gap_stop, "stop_gap", opens),
                (stop, "stop", stop_px[active]),
                (target, "target", target_px[active]),
            ):
                matched = active[hit]
                if len(matched):
                    chosen_exit[matched] = bar_index[hit]
                    exit_price[matched] = fill[hit]
                    exit_reasons[matched] = label
                    pending[matched] = False
        gross = (exit_price / ent - 1) if side == "long" else (1 - exit_price / ent)
        cost_fn = net_after_costs if side == "long" else net_after_costs_short
        net = np.array([cost_fn(float(x), fee_bps_per_side, slippage_bps_per_side)
                        for x in gross], dtype=float)
        outcomes[chosen] = (exit_reasons == "target").astype(float)
        realized_r[chosen] = net / stop_losses[chosen]
        gross_returns[chosen] = gross
        reasons[chosen] = exit_reasons
        for local, event_index in enumerate(chosen):
            entry_at[event_index] = index[first[event_index]]
            exit_at[event_index] = index[chosen_exit[local]]

    return pd.DataFrame({
        "label": outcomes,
        "net_r": realized_r,
        "gross_return": gross_returns,
        "reason": reasons,
        "entry_at": pd.to_datetime(entry_at, utc=True),
        "exit_at": pd.to_datetime(exit_at, utc=True),
        "stop_fraction": stops,
        "target_gross_fraction": target_gross,
    }, index=decisions)
