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

import pandas as pd

from research.execution import load_historical_bars
from research.funding_cost import load_funding_events, reconcile_risk_fill_funding
from research.intrabar_stress import audit_intrabar_accepted_positions
from research.shared_book_mtm import audit_shared_book_minute_mtm
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


def _validated_fold_window(report: dict, date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Trust timestamps, not a report's self-declared holdout-safe flag."""
    expected_start = pd.Timestamp(date, tz="UTC")
    expected_end = min(
        expected_start + pd.Timedelta(days=30),
        HOLDOUT_START - pd.Timedelta(days=1),  # frozen pre-holdout embargo
    )
    try:
        actual_start = pd.Timestamp(report["test_start"])
        actual_end = pd.Timestamp(report["test_end_exclusive"])
        actual_lock = pd.Timestamp(report["holdout_locked_from"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing or invalid development fold boundaries") from None
    if (any(t.tzinfo is None for t in (actual_start, actual_end, actual_lock))
        or actual_start.tz_convert("UTC") != expected_start
        or actual_end.tz_convert("UTC") != expected_end
        or actual_lock.tz_convert("UTC") != HOLDOUT_START):
        raise ValueError("development fold window or locked holdout differs from predeclared UTC boundaries")
    return expected_start, expected_end


def _validated_event_window(
    item: dict, fold_start: pd.Timestamp, fold_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A candidate must enter its declared development fold and exit pre-holdout."""
    try:
        entry = pd.Timestamp(item["entry_at"])
        exit_ = pd.Timestamp(item["exit_at"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing or invalid historical candidate timestamps") from None
    if entry.tzinfo is None or exit_.tzinfo is None:
        raise ValueError("historical candidates require timezone-aware timestamps")
    entry, exit_ = entry.tz_convert("UTC"), exit_.tz_convert("UTC")
    if (entry < fold_start or entry >= fold_end or exit_ < entry
        or exit_ >= HOLDOUT_START):
        raise ValueError("candidate crosses development entry window or locked holdout")
    return entry, exit_


def load_observed_stress_bars(root: str, symbol: str) -> pd.DataFrame:
    """Direct archived Parquet rows are observed; never bless padded rows."""
    bars = load_historical_bars(root, symbol, HOLDOUT_START)
    if "source_observed" not in bars.columns:
        bars = bars.copy()
        bars["source_observed"] = True
    return bars


def evaluate_joint(
    reports: list[dict],
    *,
    funding_resolver: Callable[[RiskFill], RiskFill] | None = None,
    stress_bars: dict | None = None,
) -> dict:
    if len(reports) != len(TARGETS) * len(STARTS):
        raise ValueError("expected 24 target-fold research reports")
    indexed = {}
    for r in reports:
        target = r.get("target")
        try:
            fold_clock = pd.Timestamp(r["test_start"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("missing or invalid development start") from None
        if fold_clock.tzinfo is None:
            raise ValueError("development start must be timezone-aware")
        date = fold_clock.tz_convert("UTC").strftime("%Y-%m-%d")
        if target not in TARGETS or date not in STARTS or (target, date) in indexed:
            raise ValueError("unknown, repeated, or out-of-window target-fold")
        _validated_fold_window(r, date)
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
        fold_start, fold_end = _validated_fold_window(indexed[(TARGETS[0], start)], start)
        candidate_exits_after_fold = 0
        for target in TARGETS:  # PREDECLARED priority for exact entry/symbol ties; never rank by future exit
            record = indexed[(target, start)]
            for item in record["joint_portfolio_candidates"]:
                if item.get("target") != target or item.get("side") != target.split("_")[1]:
                    raise ValueError("event target/direction mismatch")
                if item.get("symbol") not in ("BTCUSDT", "ETHUSDT"):
                    raise ValueError("new symbol needs separate training/audit")
                _, exit_time = _validated_event_window(item, fold_start, fold_end)
                candidate_exits_after_fold += int(exit_time >= fold_end)
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
                record_executed_positions=True,
            )
            if cost_results[name]["release_approved"] is not False:
                raise ValueError("research replay cannot approve live release")
            if funded_events is not None:
                funding_results[name] = risk_portfolio_audit(
                    funded_events, starting_capital=10000., risk_fraction=.01,
                    max_leverage=20., max_gross_exposure=6., max_concurrent_positions=2,
                    fee_bps_per_side=fee, slippage_bps_per_side=slip,
                    require_funding_reconciled=True,
                    record_executed_positions=True,
                )
                if (funding_results[name]["release_approved"] is not False or
                    (funded_events and not funding_results[name]["funding_reconciled_for_all_fills"])):
                    raise ValueError("funding adjusted research PnL must be fully reconciled")
        stress = (
            audit_intrabar_accepted_positions(
                cost_results["base"]["accepted_position_trace"],stress_bars)
            if stress_bars is not None else None
        )
        funded_stress = (
            audit_intrabar_accepted_positions(
                funding_results["base"]["accepted_position_trace"],stress_bars)
            if stress_bars is not None and funded_events is not None else None
        )
        base_mtm = (
            audit_shared_book_minute_mtm(
                cost_results["base"]["accepted_position_trace"], stress_bars)
            if stress_bars is not None else None
        )
        funded_mtm = (
            audit_shared_book_minute_mtm(
                funding_results["base"]["accepted_position_trace"], stress_bars)
            if stress_bars is not None and funded_events is not None else None
        )
        # Compare after full replay: the MTM audit must settle to precisely
        # the same historical cash as the accepted-fill shared risk ledger.
        if base_mtm is not None and not pd.isna(base_mtm["last_settled_equity"]):
            if not abs(base_mtm["last_settled_equity"] -
                       cost_results["base"]["final_equity"]) < 1.e-6:
                raise ValueError("minute MTM cash does not reconcile with shared risk book")
        if funded_mtm is not None:
            if not abs(funded_mtm["last_settled_equity"] -
                       funding_results["base"]["final_equity"]) < 1.e-6:
                raise ValueError("funding-adjusted MTM cash does not reconcile")
        # Each scenario can admit different positions as fees change;
        # completeness must therefore be audited independently per ledger.
        # Recorded exits are used only to annotate POST-replay reporting.
        for ledger in (cost_results, funding_results):
            for name, outcome in ledger.items():
                trace = outcome["accepted_position_trace"]
                late_count = sum(
                    pd.Timestamp(p["exit_at"]).tz_convert("UTC") >= fold_end
                    for p in trace
                )
                complete = late_count == 0
                outcome["accepted_positions_settling_after_fold_end"] = late_count
                outcome["net_30d_goal_assessable"] = complete
                outcome["full_entry_cohort_goal_met_after_final_settlement"] = (
                    outcome["net_30d_goal_met_in_development"]
                )
                outcome["net_30d_return_at_fold_end"] = (
                    outcome["net_equity_return"] if complete else None
                )
                outcome["net_30d_goal_met_in_development"] = (
                    outcome["net_30d_goal_met_in_development"] if complete else None
                )
                # Preserve existing API: only stress-enabled base replay
                # includes the potentially large executed-position trace.
                if stress_bars is None or name != "base":
                    outcome["accepted_position_trace"] = None

        base = cost_results["base"]
        base_late_closes = base["accepted_positions_settling_after_fold_end"]
        base_month_assessable = base["net_30d_goal_assessable"]
        funded_base = funding_results.get("base")
        funded_late_closes = (
            funded_base["accepted_positions_settling_after_fold_end"]
            if funded_base is not None else None
        )
        funded_month_assessable = (
            funded_base["net_30d_goal_assessable"]
            if funded_base is not None else None
        )
        months.append({
            "start": start,
            "base_intrabar_position_stress": stress,
            "base_minute_close_portfolio_mtm": base_mtm,
            "funding_adjusted_minute_close_portfolio_mtm": funded_mtm,
            "funding_adjusted_intrabar_position_stress": funded_stress,
            "candidate_fills": len(events),
            "candidate_exits_at_or_after_fold_end": candidate_exits_after_fold,
            "accepted_base_positions_settling_after_fold": base_late_closes,
            "base_cost_30d_goal_assessable": base_month_assessable,
            "base_cost_net_return_at_fold_end": (
                cost_results["base"]["net_equity_return"]
                if base_month_assessable else None
            ),
            "accepted_funding_positions_settling_after_fold": funded_late_closes,
            "funding_adjusted_30d_goal_assessable": funded_month_assessable,
            "funding_adjusted_net_return_at_fold_end": (
                funding_results["base"]["net_equity_return"]
                if funded_month_assessable else None
            ),
            "cost_scenarios": cost_results,
            "funding_adjusted_cost_scenarios": funding_results if funded_events is not None else None,
            "net_7pct_at_base_cost": (
                cost_results["base"]["net_30d_goal_met_in_development"]
                if base_month_assessable else None
            ),
            "net_7pct_at_base_cost_after_funding": (
                funding_results["base"]["net_30d_goal_met_in_development"]
                if funded_month_assessable else None
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
        "simultaneous_same_symbol_signal_priority": TARGETS,
        "future_exit_timestamps_used_for_admission_priority": False,
        "all_four_dev_months_net7": all(
            m["net_7pct_at_base_cost"] is True for m in months
        ),
        "all_four_dev_months_30d_goal_assessable": all(
            m["base_cost_30d_goal_assessable"] for m in months
        ),
        "holdout_untouched": True,
        "fold_entry_and_locked_holdout_windows_validated": True,
        "funding_archive_reconciliation_included": funding_resolver is not None,
        "funding_archive_not_okx_execution_data": True,
        "intrabar_ohlc_stress_audited": stress_bars is not None,
        "minute_close_shared_portfolio_mtm_audited": stress_bars is not None,
        "actual_exchange_liquidation_engine_simulated": False,
        "all_four_dev_months_net7_after_archived_funding": (
            all(m["net_7pct_at_base_cost_after_funding"] is True for m in months)
            if funding_resolver is not None else None
        ),
        "release_approved": False,
        "limitations": [
            "Shared portfolio now resolves simultaneous opportunities and margin constraints.",
            "Orders, fills and fees remain historical assumptions; no actual broker execution.",
            "Same-symbol concurrent LONG and SHORT are deliberately excluded pending hedge-mode analysis.",
            "Legacy shared portfolio settlement drawdown is realized-only; opt-in minute-close OHLC MTM is reported separately, not a liquidation engine.",
            "Intrabar stress flags possible pre-exit liquidation conservatively; not actual liquidation nor portfolio MTM.",
            "Funding-aware replay uses historical Binance-style rates, 1m open mark proxy, NOT OKX rates.",
            "Every cost scenario retains full entry-cohort realized PnL but marks its as-of-fold-end 30-day return/goal indeterminate if an accepted trade closes later.",
            "Position exits are used to annotate reporting only after the predeclared entry-priority and capacity decisions.",
            "When an accepted position is still open at fold end, no mark-to-market snapshot is available: the 30-day return and target status are unknown, not zero.",
            "Only actually accepted post-window exits invalidate the 30-day goal; rejected candidates are diagnostic only.",
            "Do not select models or tune gate settings by inspecting the locked holdout.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stress-root", default=None,
                        help="Observed archived minute OHLC for conservative accepted-position stress")
    parser.add_argument("--funding-root", default=None,
                        help="Existing historical funding archives; opt-in post-hoc funding audit")
    args = parser.parse_args()
    files = sorted(Path(args.reports_dir).glob("fishnet-y_*-dev.json"))
    report = evaluate_joint(
        [json.loads(f.read_text()) for f in files],
        funding_resolver=make_archived_funding_resolver(args.funding_root)
        if args.funding_root else None,
        stress_bars={
            sym:load_observed_stress_bars(args.stress_root,sym)
            for sym in ("BTCUSDT","ETHUSDT")
        } if args.stress_root else None,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
