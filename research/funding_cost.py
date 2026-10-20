"""Research-only realized funding adjustment from archived perpetual funding.

Unlike rolling/forward-filled funding features, this module reads settlement
EVENTS. At settlement, longs pay positive rates and shorts receive positive
rates (and vice versa for negative rates). A 1m open is only a mark-price PROXY.
It is NOT the exchange's funding mark price, order fill or live fee schedule.

Unknown rates, missing observed price minutes and unavailable funding-history
coverage fail closed. August-17-2026+ locked holdout is never read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd

LOCKED_HOLDOUT_START = pd.Timestamp("2026-08-17T00:00:00Z")
MAX_OBSERVED_EVENT_GAP = pd.Timedelta(hours=24)


@dataclass(frozen=True)
class FundingAdjustment:
    net_funding_fraction_of_entry_notional: float
    settled_funding_events: int
    payments_fraction: float
    receipts_fraction: float
    valuation_method: str = "observed_1m_open_as_mark_price_proxy"


def load_funding_events(root: str, symbol: str, *, cutoff_exclusive=LOCKED_HOLDOUT_START) -> pd.DataFrame:
    """Validate archived actual funding event timestamps; exclude locked holdout."""
    if not symbol or any(c in symbol for c in "/\\"):
        raise ValueError("invalid symbol")
    cutoff=pd.Timestamp(cutoff_exclusive)
    if cutoff.tzinfo is None or cutoff > LOCKED_HOLDOUT_START:
        raise ValueError("locked holdout cannot be read")
    paths=sorted(Path(root, symbol).glob("funding-*.parquet"))
    if not paths:
        raise ValueError("missing archived funding files")
    raw=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    if not {"calc_time","last_funding_rate"}.issubset(raw.columns):
        raise ValueError("unrecognized funding archive schema")
    millisecond=pd.to_numeric(raw.calc_time,errors="coerce")
    rates=pd.to_numeric(raw.last_funding_rate,errors="coerce").to_numpy(dtype=float)
    if (millisecond.isna().any() or not np.isfinite(rates).all()
        or np.any(np.abs(rates)>0.1)):
        raise ValueError("missing or unreasonable archived funding rate")
    timestamps=pd.to_datetime(millisecond.astype("int64"),unit="ms",utc=True)
    events=pd.DataFrame({"funding_rate":rates},index=pd.DatetimeIndex(timestamps))
    events=events.sort_index()
    if events.index.has_duplicates:
        duplicate=events.index.duplicated(keep=False)
        if (events.loc[duplicate].groupby(level=0).funding_rate.nunique()>1).any():
            raise ValueError("conflicting funding archive duplicates")
        events=events[~events.index.duplicated(keep="first")]
    events=events[events.index<cutoff]
    if events.empty:
        raise ValueError("no preholdout funding coverage")
    return events


def historical_funding_adjustment(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    side: str,
    entry_at: str | pd.Timestamp,
    exit_at: str | pd.Timestamp,
) -> FundingAdjustment:
    """Count actual archived settlement events between assumed entry and exit.

    Assumes entry is at the beginning of its minute. Funding exactly at entry
    does NOT count; funding exactly in exit minute counts (pessimistic
    ordering). Full archive coverage and no >24h reporting gaps are required.
    """
    if side not in ("long","short"):
        raise ValueError("invalid direction")
    entry,exit_=pd.Timestamp(entry_at),pd.Timestamp(exit_at)
    if (entry.tzinfo is None or exit_.tzinfo is None or entry>exit_ or
        exit_>=LOCKED_HOLDOUT_START):
        raise ValueError("invalid period or holdout exposure")
    entry=entry.tz_convert("UTC")
    exit_=exit_.tz_convert("UTC")
    if (not isinstance(events.index,pd.DatetimeIndex) or events.index.tz is None
        or not events.index.is_monotonic_increasing or events.index.has_duplicates
        or "funding_rate" not in events.columns):
        raise ValueError("validated funding events required")
    if events.empty or events.index[0]>entry or events.index[-1]<exit_:
        raise ValueError("funding history does not fully cover trade")
    # Covering sequence includes the last event before entry and the first
    # after exit, to expose missing funding-history blocks.
    start_idx=max(int(events.index.searchsorted(entry,side="right"))-1,0)
    end_idx=min(int(events.index.searchsorted(exit_,side="left"))+1,len(events)-1)
    coverage=events.index[start_idx:end_idx+1]
    if len(coverage)>1 and ((coverage[1:]-coverage[:-1])>MAX_OBSERVED_EVENT_GAP).any():
        raise ValueError("funding history has an unverified >24h gap")
    rates=events.funding_rate.to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (np.abs(rates)>.1).any():
        raise ValueError("invalid settlement rates")
    if (not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None
        or bars.index.has_duplicates or not bars.index.is_monotonic_increasing
        or "open" not in bars.columns):
        raise ValueError("observed historical minute bars required")
    try:
        entry_price=float(bars.loc[entry,"open"])
    except KeyError as exc:
        raise ValueError("missing observed entry bar") from exc
    if not np.isfinite(entry_price) or entry_price<=0:
        raise ValueError("invalid entry price")
    if "source_observed" in bars and not bool(bars.loc[entry,"source_observed"]):
        raise ValueError("unobserved entry bar")
    paid=0.0
    received=0.0
    mask=(events.index>entry)&(events.index<=exit_)
    for ts,rate in events.loc[mask,"funding_rate"].items():
        try:
            price=float(bars.loc[ts,"open"])
        except KeyError as exc:
            raise ValueError("funding settlement minute lacks observed price") from exc
        if ("source_observed" in bars and not bool(bars.loc[ts,"source_observed"])) or (
            not math.isfinite(price) or price<=0
        ):
            raise ValueError("funding settlement uses invalid or synthetic bar")
        # LONG pays a positive funding rate, SHORT receives it.
        frac=-(1 if side=="long" else -1)*float(rate)*(price/entry_price)
        if frac<0:
            paid+=-frac
        else:
            received+=frac
    return FundingAdjustment(
        net_funding_fraction_of_entry_notional=float(received-paid),
        settled_funding_events=int(mask.sum()),
        payments_fraction=float(paid),
        receipts_fraction=float(received),
    )
