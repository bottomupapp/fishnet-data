"""Conservative OHLC intratrade stress screening of ACCEPTED Fishnet positions.

Not an exchange liquidation model. A 1-minute HIGH/LOW may have occurred
AFTER a stop or target fill during the same minute, so this screening is
deliberately conservative and should be described as possible exposure,
NOT a proven historical liquidation. Does not overwrite realized portfolio
PnL or justify live orders.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from research.rr_labels import HOLDOUT_START

MAINTENANCE_MARGIN_PROXY = .005
LIQUIDATION_BUFFER_PROXY = .005


def audit_intrabar_accepted_positions(
    trace: list[dict],
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    maintenance_margin_fraction: float = MAINTENANCE_MARGIN_PROXY,
    safety_buffer_fraction: float = LIQUIDATION_BUFFER_PROXY,
) -> dict:
    """Measure worst adverse intratrade high/low on accepted positions ONLY.

    All source minutes between entry and exit (inclusive) must be observed,
    continuous, valid OHLC and strictly before locked holdout.
    """
    if any(not math.isfinite(v) or v<0 or v>=1 for v in
           (maintenance_margin_fraction, safety_buffer_fraction)):
        raise ValueError("invalid liquidation proxies")
    results=[]
    for item in trace:
        side=item.get("side")
        symbol=item.get("symbol")
        if side not in ("long","short") or symbol not in bars_by_symbol:
            raise ValueError("invalid accepted instrument or side")
        entry=pd.Timestamp(item["entry_at"])
        exit_=pd.Timestamp(item["exit_at"])
        if (entry.tzinfo is None or exit_.tzinfo is None
            or entry>exit_ or exit_>=HOLDOUT_START
            or entry.second!=0 or exit_.second!=0
            or entry.microsecond!=0 or exit_.microsecond!=0):
            raise ValueError("invalid accepted minute boundaries")
        notional=float(item["entry_notional"])
        equity=float(item["entry_equity"])
        leverage=float(item["effective_leverage"])
        stop=float(item["stop_fraction"])
        if (not all(math.isfinite(v) for v in (notional,equity,leverage,stop))
            or notional<=0 or equity<=0 or leverage<1 or not 0<stop<1):
            raise ValueError("invalid collateral or position trace")
        bars=bars_by_symbol[symbol]
        if (not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None
            or not bars.index.is_monotonic_increasing or bars.index.has_duplicates
            or not {"open","high","low","close"}.issubset(bars.columns)):
            raise ValueError("complete UTC minute OHLC data required")
        window=bars.loc[entry:exit_]
        expected=pd.date_range(entry,exit_,freq="min",tz=entry.tz)
        # Parquet bars can be datetime64[us, UTC] while generated date_range
        # is datetime64[ns, UTC]. DatetimeIndex.equals() compares the unit
        # as well as actual instants; normalize BOTH to nanoseconds in UTC.
        if not (window.index.tz_convert("UTC").as_unit("ns")
                .equals(expected.tz_convert("UTC").as_unit("ns"))):
            raise ValueError("missing observed minute inside accepted trade")
        if "source_observed" in window and not window["source_observed"].fillna(False).astype(bool).all():
            raise ValueError("synthetic OHLC during accepted trade")
        arr=window[["open","high","low","close"]].to_numpy(dtype=float)
        if (not np.isfinite(arr).all() or np.any(arr<=0)
            or np.any(arr[:,1]<arr[:,[0,2,3]].max(axis=1))
            or np.any(arr[:,2]>arr[:,[0,1,3]].min(axis=1))):
            raise ValueError("invalid price bars during accepted trade")
        entry_px=arr[0,0]
        low=float(arr[:,2].min())
        high=float(arr[:,1].max())
        adverse=max(0., (entry_px-low)/entry_px if side=="long"
                    else (high-entry_px)/entry_px)
        # Historical exchange price path and actual mark-price/liquidation
        # tiers unavailable. A rough isolated-margin distance proxy only.
        liquidation_distance=max(0.,1/leverage-maintenance_margin_fraction-safety_buffer_fraction)
        results.append({
            "symbol":symbol,"side":side,
            "entry_at":entry.isoformat(),"exit_at":exit_.isoformat(),
            "worst_adverse_ohlc_fraction":float(adverse),
            "adverse_multiple_of_planned_stop":float(adverse/stop),
            "worst_conservative_single_position_equity_excursion":
                float(adverse*notional/equity),
            "estimated_liquidation_distance_fraction":float(liquidation_distance),
            "possible_liquidation_before_reported_exit":
                bool(adverse>=liquidation_distance),
            "stop_was_touched_or_exceeded_intrabar":bool(adverse>=stop),
        })
    return {
        "accepted_position_count":len(results),
        "positions_with_possible_liquidation":sum(x["possible_liquidation_before_reported_exit"] for x in results),
        "positions_with_stop_exceedance":sum(x["stop_was_touched_or_exceeded_intrabar"] for x in results),
        "worst_single_position_equity_excursion":(
            max(x["worst_conservative_single_position_equity_excursion"] for x in results)
            if results else 0.0),
        "worst_adverse_stop_multiple":(
            max(x["adverse_multiple_of_planned_stop"] for x in results)
            if results else 0.0),
        "per_position":results,
        "portfolio_intraday_mtm_drawdown_computed":False,
        "actual_exchange_liquidation_modeled":False,
        "limitations":[
            "OHLC worst high/low within exit candle could occur after reported fill.",
            "Proxy uses traded OHLC, not exchange mark prices or true liquidation tiers.",
            "Only selected executed positions audited, not a full minute-by-minute portfolio MTM curve.",
            "Funding/liquidation fees, maintenance-tier changes, queue fills and margin call dynamics unmodeled.",
        ],
        "release_approved":False,
    }
