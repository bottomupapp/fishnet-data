"""Replay predeclared Fishnet LONG/SHORT models in ONE shared research risk book.

No hindsight ranking, no summing independent portfolio percentages, and no
holdout access. All decisions are pre-filtered by ex-ante cost gate. Simulated
fills carry hindsight outcomes only for accounting AFTER selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.risk_portfolio import RiskFill, risk_portfolio_audit

TARGETS = ("y_long_5", "y_short_5", "y_long_15", "y_short_15", "y_long_60", "y_short_60")
STARTS = ("2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17")
SCENARIOS = (("lower_cost", 3., 1.), ("base", 6., 3.), ("higher_cost", 10., 6.))


def evaluate_joint(reports: list[dict]) -> dict:
    if len(reports) != len(TARGETS) * len(STARTS):
        raise ValueError("expected 24 target-fold research reports")
    indexed = {}
    for r in reports:
        target = r.get("target")
        date = str(r.get("test_start", ""))[:10]
        if target not in TARGETS or date not in STARTS or (target, date) in indexed:
            raise ValueError("unknown, repeated, or out-of-window target-fold")
        if str(r.get("holdout_locked_from", ""))[:10] != "2026-08-17":
            raise ValueError("locked holdout changed")
        if r.get("research_abstention", {}).get("release_approved") is not False:
            raise ValueError("research gate must remain non-deployable")
        events = r.get("joint_portfolio_candidates")
        if not isinstance(events, list):
            raise ValueError("missing historical joint candidate events")
        if len(events) != r["research_abstention"]["gate_admitted"]:
            raise ValueError("candidate and gate counts differ")
        indexed[(target, date)] = r

    months = []
    for start in STARTS:
        events = []
        for target in TARGETS:  # fixed order; NOT ranked by future outcomes
            record = indexed[(target, start)]
            for item in record["joint_portfolio_candidates"]:
                if item.get("target") != target or item.get("side") != target.split("_")[1]:
                    raise ValueError("event target/direction mismatch")
                if item.get("symbol") not in ("BTCUSDT", "ETHUSDT"):
                    raise ValueError("new symbol needs separate training/audit")
                events.append(RiskFill(
                    symbol=item["symbol"], entry_at=item["entry_at"], exit_at=item["exit_at"],
                    gross_return=item["gross_return"], stop_fraction=item["stop_fraction"],
                    side=item["side"],
                ))
        cost_results = {}
        for name, fee, slip in SCENARIOS:
            cost_results[name] = risk_portfolio_audit(
                events, starting_capital=10000., risk_fraction=.01,
                max_leverage=20., max_gross_exposure=6., max_concurrent_positions=2,
                fee_bps_per_side=fee, slippage_bps_per_side=slip,
            )
            if cost_results[name]["release_approved"] is not False:
                raise ValueError("research replay cannot approve live release")
        months.append({
            "start": start,
            "candidate_fills": len(events),
            "cost_scenarios": cost_results,
            "net_7pct_at_base_cost": cost_results["base"]["net_30d_goal_met_in_development"],
        })
    return {
        "experiment": "fishnet-six-model-shared-portfolio-30d-development",
        "targets": TARGETS,
        "months": months,
        "risk_fraction_per_trade": .01,
        "shared_risk_book": True,
        "all_four_dev_months_net7": all(m["net_7pct_at_base_cost"] for m in months),
        "holdout_untouched": True,
        "release_approved": False,
        "limitations": [
            "Shared portfolio now resolves simultaneous opportunities and margin constraints.",
            "Orders, fills and fees remain historical assumptions; no actual broker execution.",
            "Same-symbol concurrent LONG and SHORT are deliberately excluded pending hedge-mode analysis.",
            "Portfolio drawdown is settlement-only; no mark-to-market liquidation or funding.",
            "Do not select models or tune gate settings by inspecting the locked holdout.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.reports_dir).glob("fishnet-y_*-dev.json"))
    report = evaluate_joint([json.loads(f.read_text()) for f in files])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
