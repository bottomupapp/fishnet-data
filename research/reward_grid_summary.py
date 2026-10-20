"""Aggregate prespecified net-R model candidates WITHOUT promoting a winner.

Twelve predeclared experiments share exactly one July development test fold.
Reports must not be aggregated into a tradeable combined portfolio: those
independent model backtests share collateral, coins, and timestamps.
Selecting a "winner" on this development window risks multiple testing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REWARDS = (1.0, 1.5, 2.0)
HORIZONS = (60, 240)
SIDES = ("long", "short")
TARGET_RETURN = .07
DEVELOPMENT_START = "2026-07-17"
HOLDOUT_LOCK = "2026-08-17"


def summarize(reports: list[dict]) -> dict:
    configs = [(side,horizon,reward) for side in SIDES for horizon in HORIZONS
               for reward in REWARDS]
    if len(reports) != len(configs):
        raise ValueError("all twelve predeclared development models are required")
    seen=set()
    rows=[]
    for item in reports:
        key=(item.get("side"),item.get("horizon_minutes"),item.get("reward_multiple_cost_net"))
        if key not in configs or key in seen:
            raise ValueError("unexpected or duplicated net-R model")
        seen.add(key)
        if (str(item.get("test_start",""))[:10] != DEVELOPMENT_START
            or str(item.get("locked_holdout_start",""))[:10] != HOLDOUT_LOCK):
            raise ValueError("test period or holdout boundary changed")
        if item.get("release_approved") is not False:
            raise ValueError("model report cannot approve production release")
        if item.get("gate_uses_calibration_probability") is not True:
            raise ValueError("uncalibrated probabilities cannot be ranked")
        if item.get("net_30d_portfolio_goal") != TARGET_RETURN:
            raise ValueError("net portfolio target changed")
        portfolio=item.get("portfolio")
        if not portfolio or portfolio.get("release_approved") is not False:
            raise ValueError("portfolio report not fail closed")
        if portfolio.get("risk_fraction_per_trade") != .01:
            raise ValueError("planned risk cap changed")
        net=portfolio.get("net_equity_return")
        if not isinstance(net,(int,float)) or not math.isfinite(net):
            raise ValueError("invalid development net return")
        if bool(portfolio["net_30d_goal_met_in_development"]) != bool(net >= TARGET_RETURN):
            raise ValueError("incorrect 7% development goal flag")
        n=item.get("economically_gated_candidate_count")
        if not isinstance(n,int) or n < 0:
            raise ValueError("invalid candidate count")
        rows.append({
            "side":key[0],"horizon_minutes":key[1],"net_reward_multiple":key[2],
            "development_net_portfolio_return":net,
            "historical_executed_fills":portfolio["executed_fills"],
            "candidate_signals":n,
            "realized_only_drawdown":portfolio["max_realized_only_drawdown"],
            "net_7pct_goal_met_in_one_development_window":net >= TARGET_RETURN,
        })
    rows.sort(key=lambda item:(item["side"],item["horizon_minutes"],item["net_reward_multiple"]))
    return {
        "experiment":"fishnet-predeclared-flexible-net-R-development-comparison",
        "development_test_start":DEVELOPMENT_START,
        "locked_holdout_start":HOLDOUT_LOCK,
        "number_of_compared_configurations":len(rows),
        "comparison_only_not_one_portfolio":True,
        "predeclared_net_rewards":REWARDS,
        "target_net_30_day_portfolio_return":TARGET_RETURN,
        "models":rows,
        "release_approved":False,
        "limitations":[
            "Only one development month is compared; consistency across months is unproven.",
            "Independent model returns must NOT be added: positions share equity and margin.",
            "This is a multiple-hypothesis comparison; winner selection needs further pre-holdout validation.",
            "Commissions/slippage are assumed; funding, liquidation, market impact and mark-to-market risk omitted.",
            "Locked holdout is untouched; no live trade or signal approval.",
        ],
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    reports=[json.loads(p.read_text()) for p in sorted(Path(args.directory).glob("fishnet-flex-r-*.json"))]
    result=summarize(reports)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
