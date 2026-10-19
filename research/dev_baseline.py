"""Safe single-fold Fishnet development baseline. Never reads locked holdout labels.
This is diagnostic ML evaluation, not a deployable signal or profitability claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from research.run import HOLDOUT_START, EMBARGO, PARAMS, feature_cols, score, dedupe
from research.execution import simulate_long, load_historical_bars, net_after_costs
from research.admission import evaluate_admission
from research.portfolio import CandidateFill, portfolio_audit
from research.risk_portfolio import RiskFill, risk_portfolio_audit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--target", default="y_long_15")
    ap.add_argument("--test-start", default="2026-07-17")
    ap.add_argument("--cost-audit-root", default=None, help="Historical bars directory; optional, research only")
    args=ap.parse_args()
    a=pd.Timestamp(args.test_start, tz="UTC")
    end=min(a+pd.Timedelta(days=30), HOLDOUT_START-EMBARGO)
    if a>=end or a>=HOLDOUT_START-EMBARGO:
        raise ValueError("test period crosses locked holdout")
    df=pd.read_parquet(args.data)
    df["t"]=pd.to_datetime(df["t"],utc=True)
    df=df[df.t < HOLDOUT_START-EMBARGO].copy()
    if args.target not in df: raise ValueError("unknown target")
    purge=pd.Timedelta(minutes=60)
    cal_start=a-EMBARGO-pd.Timedelta(days=30)
    train=df[(df.t<cal_start-EMBARGO-purge)&df[args.target].notna()]
    cal=df[(df.t>=cal_start)&(df.t<a-EMBARGO-purge)&df[args.target].notna()]
    test=df[(df.t>=a)&(df.t<end)&df[args.target].notna()]
    if min(len(train),len(cal),len(test))==0: raise ValueError("empty train/cal/test partition")
    cols=feature_cols(df)
    dataset=lgb.Dataset(train[cols], train[args.target].astype(int),free_raw_data=True)
    model=lgb.train(dict(PARAMS),dataset,num_boost_round=args.rounds)
    y=args.target
    hz=int(y.rsplit("_",1)[-1])
    test_predictions=model.predict(test[cols])
    calibration_predictions=model.predict(cal[cols])
    result=score(test,test_predictions,y,hz,calibration_predictions)
    by_symbol={}
    for symbol in sorted(test["symbol"].unique()):
        mask=(test["symbol"]==symbol).to_numpy()
        by_symbol[str(symbol)]=score(test.loc[mask],test_predictions[mask],y,hz,calibration_predictions)
    report={"target":y,"test_start":str(a),"test_end_exclusive":str(end),
            "holdout_locked_from":str(HOLDOUT_START),"rounds":args.rounds,
            "rows":{"train":len(train),"cal":len(cal),"test":len(test)},
            "features":len(cols),"metrics":result,"by_symbol":by_symbol,
            "warning":"Research-only barrier hit rates, not returns or proof of profitability. Entry at next minute open may be unrealistically optimistic without decision and routing latency; no holdout evaluation."}
    if args.cost_audit_root:
        if f"barrier_{hz}" not in df.columns:
            raise ValueError("missing training barrier for execution audit")
        # Use the same preceding-month probability threshold as the printed top-1% score.
        threshold = float(result["top0.01"]["thr"])
        selected = test.copy()
        selected["model_probability"] = test_predictions
        selected = dedupe(selected[selected["model_probability"] >= threshold], hz)
        by_symbol = {}
        all_fills = []
        admitted_fills = []
        all_portfolio_fills = []
        admitted_portfolio_fills = []
        all_risk_fills = []
        admitted_risk_fills = []
        gate_decisions = {}
        for symbol, group in selected.groupby("symbol"):
            bars = load_historical_bars(args.cost_audit_root, symbol, end)
            fills = []
            refused = {}
            for row in group.itertuples(index=False):
                admission = evaluate_admission(
                    float(row.model_probability), float(getattr(row, f"barrier_{hz}")),
                    fee_bps_per_side=6.0, slippage_bps_per_side=3.0, uncertainty_buffer_bps=5.0,
                )
                gate_decisions[admission.reason] = gate_decisions.get(admission.reason, 0) + 1
                try:
                    fill = simulate_long(
                        bars, pd.Timestamp(row.t), float(getattr(row, f"barrier_{hz}")), hz,
                        entry_delay_minutes=2, fee_bps_per_side=6, slippage_bps_per_side=3,
                    )
                    fills.append(fill)
                    all_fills.append(fill)
                    event = (str(symbol), fill)
                    all_portfolio_fills.append(event)
                    risk_event = RiskFill(
                        str(symbol), fill.entry_at, fill.exit_at, fill.gross_return,
                        float(getattr(row, f"barrier_{hz}")),
                    )
                    all_risk_fills.append(risk_event)
                    if admission.admitted:
                        admitted_fills.append(fill)
                        admitted_portfolio_fills.append(event)
                        admitted_risk_fills.append(risk_event)
                except ValueError as exc:
                    reason = str(exc)
                    refused[reason] = refused.get(reason, 0) + 1
            by_symbol[str(symbol)] = {
                "selected": int(len(group)), "simulated": len(fills),
                "refused": refused,
                "mean_net_return": float(np.mean([x.net_return for x in fills])) if fills else None,
                "net_positive_fraction": float(np.mean([x.net_return > 0 for x in fills])) if fills else None,
                "exit_reasons": {k: sum(x.exit_reason == k for x in fills)
                                 for k in ("target", "stop", "stop_gap", "time_limit")},
            }
        report["execution_audit"] = {
            "status": "illustrative_cost_assumptions_not_live_execution",
            "entry_delay_minutes_after_signal_bar_open": 2,
            "fee_bps_per_side": 6,
            "slippage_bps_per_side": 3,
            "horizon_minutes": hz,
            "selected": int(len(selected)),
            "simulated": len(all_fills),
            "mean_net_return_per_simulated_trade": (
                float(np.mean([x.net_return for x in all_fills])) if all_fills else None
            ),
            "net_positive_fraction": (
                float(np.mean([x.net_return > 0 for x in all_fills])) if all_fills else None
            ),
            "by_symbol": by_symbol,
            "limitations": "Not portfolio PnL; assumed t+2 open fills, fixed costs, no funding, partial fills or measured routing latency.",
        }
        stress = (("lower_cost", 3.0, 1.0), ("base", 6.0, 3.0), ("higher_cost", 10.0, 6.0))
        report["execution_audit"]["cost_sensitivity"] = [
            {
                "scenario": name,
                "fee_bps_per_side": fee,
                "slippage_bps_per_side": slip,
                "mean_net_return_per_simulated_trade": (
                    float(np.mean([net_after_costs(x.gross_return, fee, slip) for x in all_fills]))
                    if all_fills else None
                ),
                "net_positive_fraction": (
                    float(np.mean([net_after_costs(x.gross_return, fee, slip) > 0 for x in all_fills]))
                    if all_fills else None
                ),
            }
            for name, fee, slip in stress
        ]
        report["execution_audit"]["approved_as_trading_signal"] = False
        report["research_abstention"] = {
            "policy": "symmetric_barrier_proxy_minus_roundtrip_cost_and_5bp_uncertainty_buffer",
            "validation": "unvalidated research-only proxy; not calibrated to delayed-fill returns",
            "baseline_selected": int(len(selected)),
            "gate_admitted": gate_decisions.get("research_candidate_only", 0),
            "admitted_simulated": len(admitted_fills),
            "gate_decisions": gate_decisions,
            "mean_net_return_per_admitted_trade": (
                float(np.mean([x.net_return for x in admitted_fills]))
                if admitted_fills else None
            ),
            "cost_sensitivity": [
                {
                    "scenario": name,
                    "mean_net_return_per_admitted_trade": (
                        float(np.mean([net_after_costs(x.gross_return, fee, slip) for x in admitted_fills]))
                        if admitted_fills else None
                    ),
                }
                for name, fee, slip in stress
            ],
            "release_approved": False,
        }
        # Portfolio returns use idle cash, overlapping exposure, and compounding.
        # Cost was already deducted in AssumedFill.net_return; do not double charge.
        def equity_report(events, fee, slip):
            return portfolio_audit([
                CandidateFill(
                    symbol=symbol, entry_at=fill.entry_at, exit_at=fill.exit_at,
                    net_return=net_after_costs(fill.gross_return, fee, slip),
                ) for symbol, fill in events
            ], starting_capital=10000, stake_fraction=.10, max_concurrent_positions=2)

        report["portfolio_30d"] = {
            "window": "30-day development fold, not exact calendar month",
            "target_net_equity_return": .07,
            "model_trades_live": False,
            "ungated_cost_sensitivity": {
                name: equity_report(all_portfolio_fills, fee, slip)
                for name, fee, slip in stress
            },
            "abstention_cost_sensitivity": {
                name: equity_report(admitted_portfolio_fills, fee, slip)
                for name, fee, slip in stress
            },
            "release_approved": False,
        }
        # Leverage is only a collateral tool. Stop-distance including fees sets
        # notional; the equity-risk budget remains 1% per planned stopped trade.
        def risk_report(events, fee, slip):
            return risk_portfolio_audit(
                events, starting_capital=10000., risk_fraction=.01,
                max_leverage=20., max_gross_exposure=6.,
                max_concurrent_positions=2,
                fee_bps_per_side=fee, slippage_bps_per_side=slip,
            )

        report["risk_sized_portfolio_30d"] = {
            "model": "one_percent_equity_risk_at_cost_inclusive_stop",
            "risk_fraction_per_planned_stop": .01,
            "max_leverage": 20.,
            "max_total_gross_exposure_multiple": 6.,
            "target_net_return_per_30_day_dev_window": .07,
            "ungated_cost_sensitivity": {
                name: risk_report(all_risk_fills, fee, slip)
                for name, fee, slip in stress
            },
            "abstention_cost_sensitivity": {
                name: risk_report(admitted_risk_fills, fee, slip)
                for name, fee, slip in stress
            },
            "release_approved": False,
        }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
