import unittest
from dataclasses import replace
from research.joint_portfolio import TARGETS, STARTS, evaluate_joint

def report(target, start):
    from pandas import Timestamp, Timedelta
    at = Timestamp(start, tz="UTC") + Timedelta(days=2)
    return {
        "target": target, "test_start": start + " 00:00:00+00:00",
        "holdout_locked_from": "2026-08-17 00:00:00+00:00",
        "research_abstention": {"release_approved": False, "gate_admitted": 1},
        "joint_portfolio_candidates": [{
            "target": target, "symbol": "BTCUSDT", "side": target.split("_")[1],
            "entry_at": at.isoformat(), "exit_at": (at+Timedelta(minutes=5)).isoformat(),
            "gross_return": .01, "stop_fraction": .01,
        }],
    }

class JointBookTests(unittest.TestCase):
    def all_reports(self):
        return [report(t,s) for s in STARTS for t in TARGETS]

    def test_shared_book_does_not_add_six_independent_equity_returns(self):
        r = evaluate_joint(self.all_reports())
        first = r["months"][0]["cost_scenarios"]["base"]
        self.assertEqual(first["executed_fills"], 1)
        self.assertEqual(first["skipped_fills"]["symbol_overlap"], 5)
        self.assertFalse(r["release_approved"])

    def test_shared_funding_book_changes_realized_equity_not_signal_selection(self):
        base=evaluate_joint(self.all_reports())
        funded=evaluate_joint(
            self.all_reports(),
            funding_resolver=lambda event: replace(
                event, funding_fraction=-.001, funding_reconciled=True,
            ),
        )
        self.assertIsNone(base["months"][0]["funding_adjusted_cost_scenarios"])
        self.assertTrue(funded["funding_archive_reconciliation_included"])
        for old, new in zip(base["months"],funded["months"]):
            before=old["cost_scenarios"]["base"]
            after=new["funding_adjusted_cost_scenarios"]["base"]
            self.assertEqual(before["executed_fills"],after["executed_fills"])
            self.assertLess(after["net_equity_return"],before["net_equity_return"])
            self.assertLess(after["realized_funding_pnl_cash"],0)
            self.assertFalse(after["exchange_liquidation_engine_simulated"])
            self.assertFalse(after["release_approved"])
            self.assertLess(new["net_return_delta_from_archived_funding"],0)
        self.assertFalse(funded["release_approved"])

    def test_unreconciled_funding_replay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "funding"):
            evaluate_joint(self.all_reports(), funding_resolver=lambda fill: fill)

    def test_stress_uses_only_positions_admitted_to_single_risk_book(self):
        import numpy as np
        import pandas as pd
        parts=[]
        for start in STARTS:
            t=pd.Timestamp(start,tz="UTC")+pd.Timedelta(days=2)
            idx=pd.date_range(t,periods=10,freq="min")
            parts.append(pd.DataFrame({
                "open":np.full(10,100.),"high":np.full(10,100.1),
                "low":np.full(10,99.9),"close":np.full(10,100.),
                "source_observed":True,
            },index=idx))
        history=pd.concat(parts).sort_index()
        report=evaluate_joint(self.all_reports(),stress_bars={"BTCUSDT":history})
        self.assertTrue(report["intrabar_ohlc_stress_audited"])
        self.assertFalse(report["actual_exchange_liquidation_engine_simulated"])
        for month in report["months"]:
            stress=month["base_intrabar_position_stress"]
            self.assertEqual(stress["accepted_position_count"],1)
            self.assertEqual(stress["positions_with_possible_liquidation"],0)
            self.assertIsNone(month["funding_adjusted_intrabar_position_stress"])
            self.assertFalse(stress["portfolio_intraday_mtm_drawdown_computed"])
        self.assertFalse(report["release_approved"])

    def test_missing_reports_rejected(self):
        with self.assertRaises(ValueError): evaluate_joint(self.all_reports()[:-1])

    def test_repeated_target_fold_rejected(self):
        reports=self.all_reports()
        reports[-1]=reports[0]
        with self.assertRaises(ValueError): evaluate_joint(reports)

    def test_no_holdout_allowed(self):
        reports=self.all_reports()
        reports[0]["holdout_locked_from"]="2026-08-01"
        with self.assertRaises(ValueError): evaluate_joint(reports)

    def test_wrong_direction_rejected(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["side"]="short"
        with self.assertRaises(ValueError): evaluate_joint(reports)

    def test_gate_count_mismatch_fails(self):
        reports=self.all_reports()
        reports[0]["research_abstention"]["gate_admitted"]=2
        with self.assertRaises(ValueError): evaluate_joint(reports)

if __name__=="__main__": unittest.main()
