"""Replay predeclared Fishnet LONG/SHORT models in ONE shared research risk book.

No hindsight ranking, no summing independent portfolio percentages, and no
holdout access. All decisions are pre-filtered by ex-ante cost gate. Simulated
fills carry hindsight outcomes only for accounting AFTER selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from research.execution import load_historical_bars
from research.funding_cost import load_funding_events, reconcile_risk_fill_funding
from research.rr_labels import HOLDOUT_START

from research.risk_portfolio import RiskFill, risk_portfolio_audit

TARGETS = ("y_long_5", "y_short_5", "y_long_15", "y_short_15", "y_long_60", "y_short_60")
STARTS = ("2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17")
SCENARIOS = (("lower_cost", 3., 1.), ("base", 6., 3.), ("higher_cost", 10., 6.))


def make_archived_funding_resolver(root: str) -> Callable[[RiskFill], RiskFill]:
    """Load BTC/ETH preholdout archives once; evaluate future funding ONLY as PnL."""
    cache = {}
    def resolve(fill: RiskFill) -> RiskFill:
        if fill.symbol not in ("BTCUSDT", "ETHUSDT"):
            raise ValueError("joint funding replay supports only verified BTC/ETH universe")
        if fill.symbol not in cache:
            cache[fill.symbol] = (
                load_funding_events(root, fill.symbol, cutoff_exclusive=HOLDOUT_START),
                load_historical_bars(root, fill.symbol, HOLDOUT_START),
            )
        funding, bars = cache[fill.symbol]
        return reconcile_risk_fill_funding(fill, funding, bars)
    return resolve


def evaluate_joint(
    reports: list[dict],
    *,
    funding_resolver: Callable[[RiskFill], RiskFill] | None = None,
) -> dict:
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
        funded_events = [funding_resolver(fill) for fill in events] if funding_resolver else None
        cost_results = {}
        funding_results = {}
        for name, fee, slip in SCENARIOS:
            cost_results[name] = risk_portfolio_audit(
                events, starting_capital=10000., risk_fraction=.01,
                max_leverage=20., max_gross_exposure=6., max_concurrent_positions=2,
                fee_bps_per_side=fee, slippage_bps_per_side=slip,
            )
            if cost_results[name]["release_approved"] is not False:
                raise ValueError("research replay cannot approve live release")
            if funded_events is not None:
                funding_results[name] = risk_portfolio_audit(
                    funded_events, starting_capital=10000., risk_fraction=.01,
                    max_leverage=20., max_gross_exposure=6., max_concurrent_positions=2,
                    fee_bps_per_side=fee, slippage_bps_per_side=slip,
                    require_funding_reconciled=True,
                )
                if (funding_results[name]["release_approved"] is not False or
                    (funded_events and not funding_results[name]["funding_reconciled_for_all_fills"])):
                    raise ValueError("funding adjusted research PnL must be fully reconciled")
        months.append({
            "start": start,
            "candidate_fills": len(events),
            "cost_scenarios": cost_results,
            "funding_adjusted_cost_scenarios": funding_results if funded_events is not None else None,
            "net_7pct_at_base_cost": cost_results["base"]["net_30d_goal_met_in_development"],
            "net_7pct_at_base_cost_after_funding": (
                funding_results["base"]["net_30d_goal_met_in_development"]
                if funded_events is not None else None
            ),
            "net_return_delta_from_archived_funding": (
                funding_results["base"]["net_equity_return"] -
                cost_results["base"]["net_equity_return"]
                if funded_events is not None else None
            ),
        })
    return {
        "experiment": "fishnet-six-model-shared-portfolio-30d-development",
        "targets": TARGETS,
        "months": months,
        "risk_fraction_per_trade": .01,
        "shared_risk_book": True,
        "all_four_dev_months_net7": all(m["net_7pct_at_base_cost"] for m in months),
        "holdout_untouched": True,
        "funding_archive_reconciliation_included": funding_resolver is not None,
        "funding_archive_not_okx_execution_data": True,
        "all_four_dev_months_net7_after_archived_funding": (
            all(m["net_7pct_at_base_cost_after_funding"] for m in months)
            if funding_resolver is not None else None
        ),
        "release_approved": False,
        "limitations": [
            "Shared portfolio now resolves simultaneous opportunities and margin constraints.",
            "Orders, fills and fees remain historical assumptions; no actual broker execution.",
            "Same-symbol concurrent LONG and SHORT are deliberately excluded pending hedge-mode analysis.",
            "Portfolio drawdown is settlement-only; no mark-to-market liquidation.",
            "Funding-aware replay uses historical Binance-style rates, 1m open mark proxy, NOT OKX rates.",
            "Do not select models or tune gate settings by inspecting the locked holdout.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--funding-root", default=None,
                        help="Existing historical funding archives; opt-in post-hoc funding audit")
    args = parser.parse_args()
    files = sorted(Path(args.reports_dir).glob("fishnet-y_*-dev.json"))
    report = evaluate_joint(
        [json.loads(f.read_text()) for f in files],
        funding_resolver=make_archived_funding_resolver(args.funding_root)
        if args.funding_root else None,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
