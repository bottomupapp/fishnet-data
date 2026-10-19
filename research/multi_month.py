"""Run Fishnet's pre-holdout BTC/ETH LONG-15 development baseline across four months.

Outputs cost-adjusted research evidence; it never evaluates the locked
2026-08-17+ holdout or approves signals. The model is fitted independently
for each fold using research/dev_baseline.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FOLDS = ("2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17")
SCENARIOS = ("lower_cost", "base", "higher_cost")


def summarize(reports: list[dict]) -> dict:
    if len(reports) != len(FOLDS):
        raise ValueError("expected exactly four completed development folds")
    actual = tuple(report["test_start"][:10] for report in reports)
    if actual != FOLDS:
        raise ValueError("unexpected development fold windows")
    results = []
    for report in reports:
        ex = report.get("execution_audit")
        if not ex or ex.get("status") != "illustrative_cost_assumptions_not_live_execution":
            raise ValueError("missing conservative execution audit")
        n = int(ex["selected"])
        simulated = int(ex["simulated"])
        if n < 0 or simulated != n:
            raise ValueError("audit refused selected signals; cannot silently claim full coverage")
        scenarios = {s["scenario"]: s for s in ex["cost_sensitivity"]}
        if set(scenarios) != set(SCENARIOS):
            raise ValueError("missing expected transaction-cost scenario")
        gate = report.get("research_abstention")
        if not gate or gate.get("release_approved") is not False:
            raise ValueError("missing fail-closed research admission report")
        admitted = int(gate["gate_admitted"])
        executed = int(gate["admitted_simulated"])
        if admitted < 0 or admitted > n or executed < 0 or executed > admitted:
            raise ValueError("invalid admission counts")
        port = report.get("portfolio_30d")
        if not port or port.get("release_approved") is not False:
            raise ValueError("missing non-releasable portfolio audit")
        portfolio_summary = {}
        for name in ("ungated", "abstention"):
            scenarios_30d = port.get(f"{name}_cost_sensitivity", {})
            if set(scenarios_30d) != set(SCENARIOS):
                raise ValueError("missing portfolio cost scenarios")
            portfolio_summary[name] = {}
            for scenario in SCENARIOS:
                account = scenarios_30d[scenario]
                if account["release_approved"] is not False:
                    raise ValueError("portfolio audit cannot approve release")
                if account["target_net_30d_return"] != .07:
                    raise ValueError("portfolio goal must stay net 7 percent")
                portfolio_summary[name][scenario] = {
                    "net_equity_return": account["net_equity_return"],
                    "max_realized_drawdown": account["max_realized_drawdown"],
                    "executed_fills": account["executed_fills"],
                    "target_met": account["net_30d_goal_met_in_development"],
                }
        results.append({
            "test_start": report["test_start"],
            "test_end_exclusive": report["test_end_exclusive"],
            "test_rows": report["rows"]["test"],
            "selected_signals": n,
            "portfolio_30d": portfolio_summary,
            "research_gate": {
                "admitted": admitted,
                "simulated": executed,
                "coverage": float(admitted / n) if n else None,
                "mean_net_return": gate["mean_net_return_per_admitted_trade"],
                "decisions": gate["gate_decisions"],
            },
            "costs": {
                key: {
                    "mean_net_return": scenarios[key]["mean_net_return_per_simulated_trade"],
                    "net_positive_fraction": scenarios[key]["net_positive_fraction"],
                }
                for key in SCENARIOS
            },
        })
    gated_returns = [r["research_gate"]["mean_net_return"] for r in results]
    gated_enough = all(r["research_gate"]["simulated"] >= 20 for r in results)
    gated_consistent = gated_enough and all(v is not None and v > 0 for v in gated_returns)
    portfolio_net7_each_dev_fold = all(
        r["portfolio_30d"]["abstention"]["base"]["target_met"] for r in results
    )
    base_returns = [r["costs"]["base"]["mean_net_return"] for r in results]
    enough = all(r["selected_signals"] >= 20 for r in results)
    consistent = enough and all(v is not None and v > 0 for v in base_returns)
    return {
        "experiment": "fishnet-btc-eth-long15-fourfold-cost-audit-v1",
        "mode": "historical_development_only",
        "target": "y_long_15",
        "holdout_start": "2026-08-17T00:00:00Z",
        "folds": results,
        "development_consistency_gate_passed": consistent,
        "abstention_development_consistency_gate_passed": gated_consistent,
        "net_7pct_30d_development_goal_met_all_folds": portfolio_net7_each_dev_fold,
        "release_approved": False,
        "warning": (
            "No live approval even when the development gate passes. "
            "Four development folds are not an untouched holdout; fixed fees and slippage "
            "are hypothetical. No measured execution, funding or portfolio PnL."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("bars_root")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args()
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for start in FOLDS:
        file = output.parent / f"fishnet-{start}-dev.json"
        subprocess.run([
            sys.executable, "-m", "research.dev_baseline", args.features,
            "--output", str(file), "--rounds", str(args.rounds),
            "--target", "y_long_15", "--test-start", start,
            "--cost-audit-root", args.bars_root,
        ], check=True)
        reports.append(json.loads(file.read_text()))
    summary = summarize(reports)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
