"""Five-major *unfiltered* experimental 3.2R historical label distribution.

This does NOT train a predictive strategy, produce a trading signal, or
calculate executable portfolio profit. It measures whether cost-net +3.2R
targets can be reached before full cost-inclusive -1R stops/expiry under
pessimistic 1-minute OHLC sequencing with assumed t+2 entry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from research.execution import load_historical_bars
from research.rr_labels import HOLDOUT_START, REWARD_MULTIPLE, label_rr_events

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
FOLD_STARTS = ("2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17")
HORIZONS = (60, 240)
SIDES = ("long", "short")
EMBARGO = pd.Timedelta(days=1)


def summarize_labels(rows: pd.DataFrame, *, symbol: str, side: str, horizon: int, start: str) -> dict:
    valid = rows[rows.label.notna()]
    hits = int((valid.label == 1).sum())
    return {
        "symbol": symbol,
        "side": side,
        "horizon_minutes": horizon,
        "fold_start": start,
        "decision_rows": int(len(rows)),
        "valid_full_observation_windows": int(len(valid)),
        "invalid_gap_or_cutoff": int(len(rows) - len(valid)),
        "target_reached": hits,
        "gross_hit_fraction": float(hits / len(valid)) if len(valid) else None,
        "mean_cost_net_r_per_unfiltered_event": float(valid.net_r.mean()) if len(valid) else None,
        "median_cost_net_r_per_unfiltered_event": float(valid.net_r.median()) if len(valid) else None,
        "stop_gap_count": int((valid.reason == "stop_gap").sum()),
        "timeouts": int((valid.reason == "time_limit").sum()),
        "estimated_return_is_portfolio_pnl": False,
        "release_approved": False,
    }


def generate(root: str, feature_path: str) -> dict:
    df = pd.read_parquet(feature_path)
    if not {"symbol","t","sigma_1d"}.issubset(df.columns):
        raise ValueError("missing causal volatility features")
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[(df.t < HOLDOUT_START - EMBARGO) &
            (df.t >= pd.Timestamp(FOLD_STARTS[0], tz="UTC"))]
    if set(df.symbol.unique()) != set(SYMBOLS):
        raise ValueError("five-major research universe incomplete")
    reports=[]
    for symbol in SYMBOLS:
        market = load_historical_bars(root, symbol, HOLDOUT_START)
        # These are untouched source archive rows, so every present row was
        # observed. A missing minute is still rejected by timestamp continuity.
        market["source_observed"] = True
        features = df[df.symbol == symbol].sort_values("t").copy()
        for horizon in HORIZONS:
            stop = np.maximum(.002, features.sigma_1d.to_numpy(dtype=float) * math.sqrt(horizon))
            if not np.isfinite(stop).all():
                raise ValueError(f"non-finite {symbol} stop geometry")
            for side in SIDES:
                labels = label_rr_events(
                    market, pd.DatetimeIndex(features.t), stop,
                    side=side, horizon_minutes=horizon,
                    reward_multiple=REWARD_MULTIPLE, cutoff_exclusive=HOLDOUT_START)
                labeled = pd.DataFrame({
                    "t": pd.DatetimeIndex(features.t),
                    "label": labels.label.to_numpy(),
                    "net_r": labels.net_r.to_numpy(),
                    "reason": labels.reason.to_numpy(),
                })
                for date in FOLD_STARTS:
                    a=pd.Timestamp(date,tz="UTC")
                    b=a+pd.Timedelta(days=30)
                    selected=labeled[(labeled.t>=a) & (labeled.t<b)]
                    reports.append(summarize_labels(selected,symbol=symbol,side=side,
                                                    horizon=horizon,start=date))
    return {
        "experiment":"fishnet-unfiltered-cost-net-3.2R-label-feasibility",
        "reward_multiple_net":REWARD_MULTIPLE,
        "stop_fraction_rule":"max(0.002, sigma_1d_at_t * sqrt(horizon))",
        "assumed_fee_bps_per_side":6,
        "assumed_slippage_bps_per_side":3,
        "entry_delay_minutes":2,
        "holdout_start":HOLDOUT_START.isoformat(),
        "folds": reports,
        "model_trained":False,
        "portfolio_profit_measured":False,
        "release_approved":False,
        "warning":"Unfiltered overlapping events, NOT realized portfolio returns; assumes historical fills with no funding, measured latency, liquidation or market impact.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("bars_root")
    p.add_argument("features")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    result=generate(args.bars_root,args.features)
    Path(args.output).write_text(json.dumps(result,indent=2))
    for x in result["folds"]:
        print(f'{x["fold_start"]} {x["symbol"]} {x["side"]} {x["horizon_minutes"]}m '
              f'{x["target_reached"]}/{x["valid_full_observation_windows"]} '
              f'hit={x["gross_hit_fraction"]} mean_net_R={x["mean_cost_net_r_per_unfiltered_event"]}',flush=True)


if __name__=="__main__": main()
