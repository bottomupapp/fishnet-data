"""Fishnet 3.2R cost-net development-only LightGBM pilot.

Trains on NEW outcomes from research.rr_labels, not existing 1R barrier
labels. The locked Aug-17+ holdout is inaccessible. A predefined economic
probability hurdle applies AFTER a calibration-derived top-1% threshold.
No auto-release, live prediction feed, brokerage access or order execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

from research.execution import load_historical_bars
from research.features import HORIZONS
from research.risk_portfolio import RiskFill, risk_portfolio_audit
from research.rr_labels import HOLDOUT_START, label_rr_events, REWARD_MULTIPLE
from research.run import PARAMS, feature_cols, dedupe


DEVELOPMENT_TEST_START = pd.Timestamp("2026-07-17T00:00:00Z")
EMBARGO = pd.Timedelta(days=1)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
ECONOMIC_PROBABILITY_FLOOR = 1.0 / (1.0 + REWARD_MULTIPLE)


def partition_boundaries(test_start: pd.Timestamp, horizon: int) -> dict[str, pd.Timestamp]:
    """Purges the entire label outcome window plus an extra 24-hour embargo."""
    test_start = pd.Timestamp(test_start)
    if test_start.tzinfo is None or not 1 <= horizon <= 240:
        raise ValueError("invalid development start or horizon")
    test_end = min(test_start + pd.Timedelta(days=30), HOLDOUT_START - EMBARGO)
    if test_start >= test_end or test_start >= HOLDOUT_START - EMBARGO:
        raise ValueError("test overlaps locked holdout or embargo")
    purge = pd.Timedelta(minutes=horizon + 2)
    cal_start = test_start - pd.Timedelta(days=30) - EMBARGO
    return {
        "train_end_exclusive": cal_start - EMBARGO - purge,
        "cal_start": cal_start,
        "cal_end_exclusive": test_start - EMBARGO - purge,
        "test_start": test_start,
        "test_end_exclusive": test_end,
    }


def safe_model_columns(df: pd.DataFrame) -> list[str]:
    """Freeze causal feature list before attaching any realized target outcomes."""
    columns = feature_cols(df)
    forbidden = ("gross_return", "net_r", "rr_label", "rr_stop_fraction", "rr_entry_at", "rr_exit_at")
    if any(c in columns for c in forbidden) or any(c.startswith("rr_") for c in columns):
        raise ValueError("outcome or replay metadata cannot enter LightGBM features")
    if not columns:
        raise ValueError("no model features")
    return columns


def run(features_file: str, root: str, *, side: str, horizon: int, rounds: int=100) -> dict:
    if side not in ("long", "short") or horizon not in (60,240) or rounds < 1:
        raise ValueError("invalid 3.2R development configuration")
    df = pd.read_parquet(features_file)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[(df.t >= pd.Timestamp("2025-10-18T00:00:00Z")) &
            (df.t < HOLDOUT_START - EMBARGO)].copy()
    if set(df.symbol.unique()) != set(SYMBOLS):
        raise ValueError("five-coin universe incomplete")
    if df.duplicated(["symbol","t"]).any():
        raise ValueError("duplicate feature event")
    columns = safe_model_columns(df)
    df["rr_label"] = np.nan
    df["rr_net_r"] = np.nan
    df["rr_gross_return"] = np.nan
    df["rr_stop_fraction"] = np.nan
    df["rr_entry_at"] = pd.NaT
    df["rr_exit_at"] = pd.NaT
    for symbol in SYMBOLS:
        rows = df[df.symbol == symbol].sort_values("t")
        market = load_historical_bars(root, symbol, HOLDOUT_START)
        market["source_observed"] = True  # raw archive rows; gaps caught by index checks
        stops = np.maximum(.002, rows.sigma_1d.to_numpy(dtype=float) * math.sqrt(horizon))
        if not np.isfinite(stops).all():
            raise ValueError(f"bad prior volatility for {symbol}")
        outcomes = label_rr_events(
            market, pd.DatetimeIndex(rows.t), stops,
            side=side, horizon_minutes=horizon, cutoff_exclusive=HOLDOUT_START)
        df.loc[rows.index,"rr_label"] = outcomes.label.to_numpy()
        df.loc[rows.index,"rr_net_r"] = outcomes.net_r.to_numpy()
        df.loc[rows.index,"rr_gross_return"] = outcomes.gross_return.to_numpy()
        df.loc[rows.index,"rr_stop_fraction"] = stops
        df.loc[rows.index,"rr_entry_at"] = outcomes.entry_at.to_numpy()
        df.loc[rows.index,"rr_exit_at"] = outcomes.exit_at.to_numpy()

    w = partition_boundaries(DEVELOPMENT_TEST_START, horizon)
    train = df[(df.t < w["train_end_exclusive"]) & df.rr_label.notna()]
    cal = df[(df.t >= w["cal_start"]) & (df.t < w["cal_end_exclusive"]) & df.rr_label.notna()]
    test = df[(df.t >= w["test_start"]) & (df.t < w["test_end_exclusive"]) & df.rr_label.notna()]
    if any(len(x) == 0 for x in (train, cal, test)):
        raise ValueError("empty development train/calibration/test partition")
    if int(train.rr_label.sum()) < 30 or int(cal.rr_label.sum()) < 5:
        raise ValueError("too few 3.2R target positives for exploratory fit")
    model = lgb.train(dict(PARAMS), lgb.Dataset(train[columns],
        train.rr_label.astype(int)), num_boost_round=rounds)
    cal_scores = model.predict(cal[columns])
    test_scores = model.predict(test[columns])
    cal_cutoff = float(np.quantile(cal_scores, .99))
    test_with_p = test.copy()
    test_with_p["prediction"] = test_scores
    top = dedupe(test_with_p[test_with_p.prediction >= cal_cutoff], horizon)
    gate_threshold = max(cal_cutoff, ECONOMIC_PROBABILITY_FLOOR)
    gated = dedupe(test_with_p[test_with_p.prediction >= gate_threshold], horizon)

    fills = [
        RiskFill(str(row.symbol), pd.Timestamp(row.rr_entry_at).isoformat(),
                 pd.Timestamp(row.rr_exit_at).isoformat(),
                 float(row.rr_gross_return), float(row.rr_stop_fraction), side)
        for row in gated.itertuples(index=False)
    ]
    portfolio = risk_portfolio_audit(
        fills, starting_capital=10000., risk_fraction=.01, max_leverage=20.,
        max_gross_exposure=6., max_concurrent_positions=2,
        fee_bps_per_side=6., slippage_bps_per_side=3.,
    )
    return {
        "experiment":"fishnet-32r-directional-ml-single-development-fold",
        "side":side, "horizon_minutes":horizon,
        "reward_multiple_cost_net":REWARD_MULTIPLE,
        "symbols":SYMBOLS,
        "test_start":w["test_start"].isoformat(),
        "test_end_exclusive":w["test_end_exclusive"].isoformat(),
        "locked_holdout_start":HOLDOUT_START.isoformat(),
        "train_rows":len(train), "calibration_rows":len(cal), "test_rows":len(test),
        "train_target_hit_count":int(train.rr_label.sum()),
        "cal_target_hit_count":int(cal.rr_label.sum()),
        "test_unfiltered_hit_fraction":float(test.rr_label.mean()),
        "feature_count":len(columns),
        "calibration_top1_cutoff":cal_cutoff,
        "economic_1R_loss_3_2R_gain_probability_floor":ECONOMIC_PROBABILITY_FLOOR,
        "gate_probability_threshold":gate_threshold,
        "calibrated_top1_candidate_count":int(len(top)),
        "calibrated_top1_historical_hit_fraction":float(top.rr_label.mean()) if len(top) else None,
        "calibrated_top1_mean_cost_net_R":float(top.rr_net_r.mean()) if len(top) else None,
        "economically_gated_candidate_count":int(len(gated)),
        "economically_gated_historical_mean_cost_net_R":
            float(gated.rr_net_r.mean()) if len(gated) else None,
        "portfolio":portfolio,
        "release_approved":False,
        "limitations": [
            "Single development fold, not an untouched holdout or real execution.",
            "Probability floor assumes winners +3.2R and losers -1R; timeouts and gaps differ.",
            "Model calibration beyond quantile thresholds is unproven; no production signals.",
            "Costs and margin are assumed; no funding, true liquidation, partial fills or impact.",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("features")
    p.add_argument("bars_root")
    p.add_argument("--side",required=True,choices=("long","short"))
    p.add_argument("--horizon",required=True,type=int,choices=(60,240))
    p.add_argument("--rounds",type=int,default=100)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    result=run(args.features,args.bars_root,side=args.side,horizon=args.horizon,rounds=args.rounds)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
