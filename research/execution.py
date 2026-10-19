"""Conservative, offline LONG execution audit for Fishnet research.

Research only: results are *assumed* fills and NEVER executable orders.
A signal after minute t closes first enters at t+2 open by default; this
deliberately avoids the unrealistically instantaneous t+1 open assumption.
Stops take precedence when the same 1-minute bar touches both barriers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssumedFill:
    entry_at: str
    exit_at: str
    exit_reason: str
    gross_return: float
    net_return: float


def simulate_long(
    bars: pd.DataFrame,
    decision_at: pd.Timestamp,
    barrier_fraction: float,
    horizon_minutes: int,
    *,
    entry_delay_minutes: int = 2,
    fee_bps_per_side: float = 6.0,
    slippage_bps_per_side: float = 3.0,
) -> AssumedFill:
    """Bracket long trade with pessimistic OHLC sequencing and explicit costs.

    Bar timestamps are UTC minute *opens*. The decision is made only after
    decision_at's bar has closed. t+2 entry is still an *assumed* fill, not a
    latency-measured execution. Missing bars or market gaps fail closed.
    """
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("UTC-aware minute index required")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("minute index must be sorted and unique")
    if not 0 < barrier_fraction < 1 or not np.isfinite(barrier_fraction):
        raise ValueError("barrier_fraction must be finite and between zero and one")
    if horizon_minutes < 1 or entry_delay_minutes < 2:
        raise ValueError("horizon must be positive and entry delay must be at least two minute bars")
    if not all(np.isfinite(v) and v >= 0 for v in (fee_bps_per_side, slippage_bps_per_side)):
        raise ValueError("invalid execution cost")
    if fee_bps_per_side >= 10000 or slippage_bps_per_side >= 10000:
        raise ValueError("execution cost must be below 100%")
    required = {"open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError("missing OHLC fields")
    ts = pd.Timestamp(decision_at)
    if ts.tzinfo is None:
        raise ValueError("decision_at must be timezone-aware")
    try:
        idx = bars.index.get_loc(ts)
    except KeyError as exc:
        raise ValueError("decision timestamp absent from historical bars") from exc
    if not isinstance(idx, (int, np.integer)):
        raise ValueError("ambiguous decision timestamp")
    begin = int(idx) + entry_delay_minutes
    end = begin + horizon_minutes
    if end > len(bars):
        raise ValueError("insufficient observed future bars")
    period = bars.iloc[int(idx):end]
    expected = pd.date_range(ts, periods=entry_delay_minutes + horizon_minutes, freq="min", tz=ts.tz)
    if not period.index.equals(expected):
        raise ValueError("missing minute inside simulation window")
    if "source_observed" in period and not period["source_observed"].fillna(False).astype(bool).all():
        raise ValueError("synthetic or missing source bar inside window")
    ohlc = period[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        raise ValueError("invalid OHLC prices")
    if ((ohlc[:, 1] < ohlc[:, [0, 2, 3]].max(axis=1)) |
        (ohlc[:, 2] > ohlc[:, [0, 1, 3]].min(axis=1))).any():
        raise ValueError("inconsistent OHLC bars")

    entry = float(bars.iloc[begin]["open"])
    target, stop = entry * (1 + barrier_fraction), entry * (1 - barrier_fraction)
    exit_px, reason, exit_idx = float(bars.iloc[end - 1]["close"]), "time_limit", end - 1
    for k in range(begin, end):
        bar = bars.iloc[k]
        # At the bar open, a gap through the stop executes at the *worse* open.
        if float(bar["open"]) <= stop:
            exit_px, reason, exit_idx = float(bar["open"]), "stop_gap", k
            break
        if float(bar["low"]) <= stop:
            exit_px, reason, exit_idx = stop, "stop", k
            break
        # TP fill capped at limit target even when bar opens better.
        if float(bar["high"]) >= target:
            exit_px, reason, exit_idx = target, "target", k
            break
    fee, slip = fee_bps_per_side / 10000, slippage_bps_per_side / 10000
    gross = exit_px / entry - 1
    net = (exit_px * (1 - slip) * (1 - fee)) / (entry * (1 + slip) * (1 + fee)) - 1
    return AssumedFill(
        entry_at=bars.index[begin].isoformat(),
        exit_at=bars.index[exit_idx].isoformat(),
        exit_reason=reason,
        gross_return=float(gross),
        net_return=float(net),
    )


def load_historical_bars(root: str, symbol: str, cutoff_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Limit the audit to pre-cutoff periods; never simulate locked holdout."""
    cutoff = pd.Timestamp(cutoff_exclusive)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    month = (cutoff - pd.Timedelta(minutes=1)).strftime("%Y-%m")
    paths = sorted(p for p in Path(root, symbol).glob("20??-??.parquet") if p.stem <= month)
    if not paths:
        raise ValueError("no eligible historical bar files")
    df = pd.concat(pd.read_parquet(p) for p in paths).sort_index()
    if df.index.tz is None:
        raise ValueError("historical timestamps lack time zone")
    df = df[df.index < cutoff]
    if df.index.has_duplicates:
        raise ValueError("duplicate minute timestamps")
    return df


def net_after_costs(gross_return: float, fee_bps_per_side: float, slippage_bps_per_side: float) -> float:
    """Apply symmetric fixed-cost stress to an already simulated gross return."""
    values = (gross_return, fee_bps_per_side, slippage_bps_per_side)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("non-finite return or execution cost")
    if gross_return <= -1 or not (0 <= fee_bps_per_side < 10000) or not (0 <= slippage_bps_per_side < 10000):
        raise ValueError("invalid gross return or execution cost")
    fee, slip = fee_bps_per_side / 10000, slippage_bps_per_side / 10000
    return float((1 + gross_return) * (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee)) - 1)
