"""Aggregate six distinct Fishnet development experiments without promotion.

All six targets were pre-specified; the locked holdout remains untouched.
A development leaderboard is NOT proof of out-of-sample predictive edge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = (
    "y_long_5", "y_short_5",
    "y_long_15", "y_short_15",
    "y_long_60", "y_short_60",
)
FOLD_STARTS = ("2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17")


def aggregate(reports: list[dict]) -> dict:
    if len(reports) != len(TARGETS):
        raise ValueError("exactly six target reports are required")
    by_target = {}
    for report in reports:
        target = report.get("target")
        if target not in TARGETS or target in by_target:
            raise ValueError("unknown or duplicated target")
        if report.get("release_approved") is not False:
            raise ValueError("a research report must never approve live trading")
        if report.get("holdout_start") != "2026-08-17T00:00:00Z":
            raise ValueError("locked holdout was changed")
        folds = report.get("folds", [])
        if len(folds) != len(FOLD_STARTS):
            raise ValueError("all four development folds must exist")
        actual = tuple(str(f["test_start"])[:10] for f in folds)
        if actual != FOLD_STARTS:
            raise ValueError("unexpected or reordered fold")
        rows = []
        for f in folds:
            entry = f["risk_sized_portfolio_30d"]["abstention"]["base"]
            net = entry["net_equity_return"]
            trades = entry["executed_fills"]
            dd = entry["max_realized_only_drawdown"]
            if not isinstance(net, (int, float)) or not -1 < net < 100:
                raise ValueError("invalid net portfolio return")
            if not isinstance(trades, int) or trades < 0 or not 0 <= dd <= 1:
                raise ValueError("invalid capacity or drawdown")
            rows.append({
                "start": str(f["test_start"])[:10],
                "net_30day_equity_return": net,
                "simulated_fills": trades,
                "realized_only_drawdown": dd,
                "mean_realized_r": entry["mean_realized_r"],
                "net_7pct_goal_met": bool(entry["target_met"]),
            })
        # The research candidate filter is predeclared; no threshold selection
        # is based on outcomes from the same fold.
        four_positive = all(x["net_30day_equity_return"] > 0 for x in rows)
        four_above_goal = all(x["net_7pct_goal_met"] for x in rows)
        enough = all(x["simulated_fills"] >= 20 for x in rows)
        by_target[target] = {
            "target": target,
            "folds": rows,
            "total_simulated_fills": sum(x["simulated_fills"] for x in rows),
            "positive_in_all_dev_folds": four_positive,
            "above_net_7pct_in_all_dev_folds": four_above_goal,
            "min_20_simulated_fills_per_dev_fold": enough,
            "research_consistency_passed": four_above_goal and enough,
            "release_approved": False,
        }
    if set(by_target) != set(TARGETS):
        raise ValueError("missing target")
    return {
        "title": "Fishnet LONG/SHORT 5/15/60 development comparison",
        "mode": "historical_research_only",
        "targets": [by_target[t] for t in TARGETS],
        "evaluated_distinct_targets": len(TARGETS),
        "net_7pct_goal": .07,
        "locked_holdout_untouched": True,
        "release_approved": False,
        "limitations": [
            "Multiple development hypotheses are evaluated; rankings can overfit development folds.",
            "Target selection must be frozen before using the independent holdout, and must not be optimized on holdout results.",
            "Assumed fees/slippage and liquidation proxy are not verified exchange execution.",
            "Realized-only drawdown is optimistic; funding, taxes, partial fills, and market impact are excluded.",
            "No model is approved for live signals or order execution.",
        ],
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("reports_directory")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    root=Path(args.reports_directory)
    paths=sorted(root.glob("fishnet-y_*-grid.json"))
    reports=[json.loads(p.read_text()) for p in paths]
    results=aggregate(reports)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
