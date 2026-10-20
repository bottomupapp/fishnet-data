"""Fishnet exploratory LightGBM net-R regression, NOT barrier-hit classification.

Goal: predict direction-specific realized NET R including timeouts and stop gaps.
Fits a regression target only on the development training window, selects a
prediction cutoff and tests mean realized net-R only on a PRE-test calibration
month. Test labels never select candidates. Positive calibration evidence
does not approve live deployment; monthly 7% remains a portfolio benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.execution import load_historical_bars
from research.rr_labels import HOLDOUT_START, label_rr_events
from research.rr_model import SYMBOLS, DEVELOPMENT_TEST_START, partition_boundaries, safe_model_columns
from research.net_r_gate import calibrate_r_gate
from research.regime_gate import fit_regime_gates, select_from_frozen_gates
from research.risk_portfolio import RiskFill, risk_portfolio_audit
from research.run import PARAMS, dedupe

PREDECLARED_REWARDS = (1.5,)
PREDECLARED_HORIZONS = (60, 240)
ROUND_TRIP_FEE_BPS_PER_SIDE = 6.
SLIPPAGE_BPS_PER_SIDE = 3.


def run(features_file: str, root: str, *, side: str, horizon: int, reward: float=1.5,
        rounds: int=100) -> dict:
    if (side not in ("long","short") or horizon not in PREDECLARED_HORIZONS
        or reward not in PREDECLARED_REWARDS or rounds < 1):
        raise ValueError("unapproved exploratory target configuration")

    df=pd.read_parquet(features_file)
    df["t"]=pd.to_datetime(df["t"],utc=True)
    df=df[df.t < HOLDOUT_START-pd.Timedelta(days=1)].copy()
    if set(df.symbol.unique()) != set(SYMBOLS) or df.duplicated(["symbol","t"]).any():
        raise ValueError("incomplete or duplicate five-major research data")
    features=safe_model_columns(df)
    for col in ("net_r","event_gross_return","event_stop_fraction"):
        df[col]=np.nan
    for col in ("event_entry","event_exit"):
        df[col]=None
    for symbol in SYMBOLS:
        rows=df[df.symbol==symbol].sort_values("t")
        market=load_historical_bars(root,symbol,HOLDOUT_START)
        market["source_observed"]=True
        stops=np.maximum(.002, rows.sigma_1d.to_numpy(dtype=float)*math.sqrt(horizon))
        if not np.isfinite(stops).all():
            raise ValueError("invalid historical volatility")
        events=label_rr_events(market,pd.DatetimeIndex(rows.t),stops,side=side,
                               horizon_minutes=horizon,reward_multiple=reward,
                               cutoff_exclusive=HOLDOUT_START)
        df.loc[rows.index,"net_r"]=events.net_r.to_numpy()
        df.loc[rows.index,"event_gross_return"]=events.gross_return.to_numpy()
        df.loc[rows.index,"event_stop_fraction"]=stops
        df.loc[rows.index,"event_entry"]=[
            t.isoformat() if pd.notna(t) else None for t in events.entry_at
        ]
        df.loc[rows.index,"event_exit"]=[
            t.isoformat() if pd.notna(t) else None for t in events.exit_at
        ]

    windows=partition_boundaries(DEVELOPMENT_TEST_START,horizon)
    train=df[(df.t < windows["train_end_exclusive"]) & df.net_r.notna()]
    cal=df[(df.t >= windows["cal_start"]) & (df.t < windows["cal_end_exclusive"])
           & df.net_r.notna()]
    test=df[(df.t >= windows["test_start"]) & (df.t < windows["test_end_exclusive"])
            & df.net_r.notna()]
    if any(len(v)<100 for v in (train,cal,test)):
        raise ValueError("empty or insufficient research split")

    params=dict(PARAMS)
    params.update(objective="regression",metric="l2")
    # Winsorization protects the FIT from extreme gap outliers. Portfolio
    # accounting still uses untouched historical gross/net outcomes.
    model=lgb.train(params,lgb.Dataset(train[features],
        np.clip(train.net_r.to_numpy(dtype=float),-4,4)),num_boost_round=rounds)
    cal_p=model.predict(cal[features])
    test_p=model.predict(test[features])
    gate=calibrate_r_gate(cal,cal_p,horizon_minutes=horizon)

    test_with_prediction=test.copy()
    test_with_prediction["expected_r_model_score"]=test_p
    top=dedupe(test_with_prediction[test_with_prediction.expected_r_model_score >=
                                  gate.score_threshold],horizon)
    admitted=top if gate.admitted_for_research else top.iloc[:0].copy()

    # Second, strictly predeclared research arm: trend-direction × trailing
    # relative volatility, with a low-activity exclusion. Every decision
    # is frozen using realized R from the earlier calibration month ONLY.
    regime_evidence=fit_regime_gates(cal,cal_p,side=side,horizon_minutes=horizon)
    regime_admitted=select_from_frozen_gates(
        test,test_p,regime_evidence,side=side,horizon_minutes=horizon,
    )
    regime_fills=[
        RiskFill(str(row.symbol),str(row.event_entry),str(row.event_exit),
                 float(row.event_gross_return),float(row.event_stop_fraction),side)
        for row in regime_admitted.itertuples(index=False)
    ]
    regime_portfolio=risk_portfolio_audit(
        regime_fills,starting_capital=10000.,risk_fraction=.01,
        max_leverage=20.,max_gross_exposure=6.,max_concurrent_positions=2,
        fee_bps_per_side=ROUND_TRIP_FEE_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    )
    fills=[
        RiskFill(str(row.symbol), str(row.event_entry), str(row.event_exit),
                 float(row.event_gross_return),float(row.event_stop_fraction),side)
        for row in admitted.itertuples(index=False)
    ]
    portfolio=risk_portfolio_audit(
        fills,starting_capital=10000.,risk_fraction=.01,
        max_leverage=20.,max_gross_exposure=6.,max_concurrent_positions=2,
        fee_bps_per_side=ROUND_TRIP_FEE_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    )
    return {
        "experiment":"fishnet-development-expected-net-R-pilot",
        "side":side,"horizon_minutes":horizon,"reward_multiple_net":reward,
        "symbols":SYMBOLS,
        "test_start":windows["test_start"].isoformat(),
        "test_end_exclusive":windows["test_end_exclusive"].isoformat(),
        "holdout_start":HOLDOUT_START.isoformat(),
        "train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),
        "features":len(features),
        "training_target":"winsorized realized cost-net R including timeouts",
        "top_decile_calibration":asdict(gate),
        "test_prediction_top_decile":int(len(top)),
        "test_preselected_mean_net_r":float(top.net_r.mean()) if len(top) else None,
        "admitted_candidate_count":int(len(admitted)),
        "portfolio":portfolio,
        "regime_research": {
            "policy": "4_predeclared_trend_direction_by_relative_volatility_low_activity_excluded",
            "gate_fitted_only_on_pretest_calibration": True,
            "volume_z_is_not_orderbook_liquidity": True,
            "evidence": {name:asdict(e) for name,e in regime_evidence.items()},
            "admitted_candidate_count":len(regime_admitted),
            "admitted_historical_mean_net_r":(
                float(regime_admitted.net_r.mean()) if len(regime_admitted) else None
            ),
            "portfolio":regime_portfolio,
            "release_approved":False,
        },
        "goal_7pct_net_per_development_window_achieved":
            portfolio["net_30d_goal_met_in_development"],
        "release_approved":False,
        "limitations":[
            "Single development month; no independent holdout validation.",
            "Prediction is not verified expected net-R or a guaranteed return.",
            "Non-overlapping calibration removes within-symbol overlaps, not cross-coin correlations.",
            "Four regime evaluations on one calibration month create multiple testing risk.",
            "Activity z-score is a proxy and not a measurement of order-book depth.",
            "Two-standard-error calibration haircut is heuristic, not a formal confidence interval.",
            "Assumed costs and liquidation proxy; no measured fills, funding or mark-to-market risk.",
            "Historical strategy is research-only, not real trading or executable orders.",
        ],
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("bars_root")
    ap.add_argument("--side",required=True,choices=("long","short"))
    ap.add_argument("--horizon",required=True,type=int,choices=PREDECLARED_HORIZONS)
    ap.add_argument("--reward",type=float,default=1.5,choices=PREDECLARED_REWARDS)
    ap.add_argument("--rounds",type=int,default=100)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    result=run(args.features,args.bars_root,side=args.side,horizon=args.horizon,
               reward=args.reward,rounds=args.rounds)
    output=Path(args.output)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
