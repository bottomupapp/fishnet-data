"""Verify real funding archive coverage and cashflow direction on five majors.

Research-only settlement audit, NOT a trading strategy or profitability report.
The July example is a predetermined 4-hour holding period crossing 08:00 UTC.
Funding valuation uses observed 1m OPEN as an approximate mark, not real
exchange mark price. No holdout data, model selection or live orders.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.funding_cost import (
    LOCKED_HOLDOUT_START, MAX_OBSERVED_EVENT_GAP,
    historical_funding_adjustment, load_funding_events
)

SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT")
START=pd.Timestamp("2026-07-17T00:00:00Z")
END=pd.Timestamp("2026-08-16T12:00:00Z")
EXAMPLE_ENTRY=pd.Timestamp("2026-07-21T06:00:00Z")
EXAMPLE_EXIT=pd.Timestamp("2026-07-21T10:00:00Z")


def audit(root: str) -> dict:
    rows=[]
    for symbol in SYMBOLS:
        events=load_funding_events(root,symbol)
        if events.index[0]>START or events.index[-1]<END:
            raise ValueError(f"{symbol}: missing full preholdout funding history")
        coverage=events.loc[START:END]
        if coverage.empty:
            raise ValueError(f"{symbol}: no funding events in evaluation window")
        if len(coverage)>1 and ((coverage.index[1:]-coverage.index[:-1])>
                                MAX_OBSERVED_EVENT_GAP).any():
            raise ValueError(f"{symbol}: unverified funding event gap")
        path=Path(root,symbol,"2026-07.parquet")
        if not path.is_file():
            raise ValueError(f"{symbol}: missing July observed minute bars")
        bars=pd.read_parquet(path).sort_index()
        if bars.index.tz is None or bars.index.has_duplicates:
            raise ValueError(f"{symbol}: bad July bar timestamps")
        sample_long=historical_funding_adjustment(
            events,bars,side="long",entry_at=EXAMPLE_ENTRY,exit_at=EXAMPLE_EXIT)
        sample_short=historical_funding_adjustment(
            events,bars,side="short",entry_at=EXAMPLE_ENTRY,exit_at=EXAMPLE_EXIT)
        if not np.isclose(sample_long.net_funding_fraction_of_entry_notional,
                          -sample_short.net_funding_fraction_of_entry_notional):
            raise ValueError("directional funding cashflows do not balance")
        rows.append({
            "symbol":symbol,
            "funding_event_count":int(len(coverage)),
            "positive_rate_count":int((coverage.funding_rate>0).sum()),
            "negative_rate_count":int((coverage.funding_rate<0).sum()),
            "min_archived_funding_rate":float(coverage.funding_rate.min()),
            "max_archived_funding_rate":float(coverage.funding_rate.max()),
            "last_preholdout_event":events.index[-1].isoformat(),
            "example_entry":EXAMPLE_ENTRY.isoformat(),
            "example_exit":EXAMPLE_EXIT.isoformat(),
            "example_funding_events":sample_long.settled_funding_events,
            "example_long_net_funding_fraction":
                sample_long.net_funding_fraction_of_entry_notional,
            "example_short_net_funding_fraction":
                sample_short.net_funding_fraction_of_entry_notional,
        })
    return {
        "experiment":"fishnet-five-major-realized-funding-archive-validation-v1",
        "start":START.isoformat(),
        "end_inclusive":END.isoformat(),
        "holdout_untouched":True,
        "fee_schedule_not_verified":True,
        "funding_uses_1m_open_mark_proxy":True,
        "historical_sample_is_not_portfolio_pnl":True,
        "exchange_liquidation_engine_simulated":False,
        "release_approved":False,
        "coins":rows,
        "limitations":[
            "Historical archived settlements are never known in advance at decision time.",
            "Open prices are NOT actual exchange funding mark prices.",
            "Historical settlement timestamps/rates do not validate a user's exchange fee tier.",
            "No partial fills, funding-mark prices, broker margin tiers or liquidation engine.",
        ],
    }


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=audit(args.root)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
