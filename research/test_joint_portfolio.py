import unittest
from dataclasses import replace
from research.joint_portfolio import TARGETS, STARTS, evaluate_joint

def report(target, start):
    from pandas import Timestamp, Timedelta
    at = Timestamp(start, tz="UTC") + Timedelta(days=2)
    return {
        "target": target, "test_start": start + " 00:00:00+00:00",
        "test_end_exclusive": min(
            Timestamp(start, tz="UTC") + Timedelta(days=30),
            Timestamp("2026-08-16T00:00:00Z"),
        ).isoformat(),
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

    def test_direct_archive_rows_are_explicitly_observed_for_mtm(self):
        from unittest.mock import patch
        import pandas as pd
        from research.joint_portfolio import load_observed_stress_bars
        idx=pd.date_range("2026-07-01T00:00:00Z",periods=2,freq="min")
        raw=pd.DataFrame({"open":[100.,101.],"high":[101.,102.],
                          "low":[99.,100.],"close":[100.,101.]},index=idx)
        with patch("research.joint_portfolio.load_historical_bars",return_value=raw):
            marked=load_observed_stress_bars("data","BTCUSDT")
        self.assertTrue(marked["source_observed"].all())
        self.assertNotIn("source_observed",raw.columns)

    def test_source_archive_explicit_missing_rows_never_overridden(self):
        from unittest.mock import patch
        import pandas as pd
        from research.joint_portfolio import load_observed_stress_bars
        idx=pd.date_range("2026-07-01T00:00:00Z",periods=2,freq="min")
        raw=pd.DataFrame({"source_observed":[True,False]},index=idx)
        with patch("research.joint_portfolio.load_historical_bars",return_value=raw):
            marked=load_observed_stress_bars("data","BTCUSDT")
        self.assertEqual(marked["source_observed"].tolist(),[True,False])

    def test_single_accepted_fill_cannot_claim_robust_monthly_evidence(self):
        r=evaluate_joint(self.all_reports())
        self.assertEqual(r["minimum_accepted_fills_per_development_fold"],20)
        self.assertFalse(r["all_four_dev_months_have_sufficient_accepted_fills"])
        self.assertFalse(r["all_four_dev_months_net7_with_minimum_sample"])
        for m in r["months"]:
            self.assertEqual(m["base_accepted_fills"],1)
            self.assertFalse(m["base_has_sufficient_accepted_fill_count"])
        self.assertFalse(r["release_approved"])

    def test_predeclared_sample_floor_does_not_change_trade_admission(self):
        reports=self.all_reports()
        first=evaluate_joint(reports)
        second=evaluate_joint(reports)
        self.assertEqual(
            [m["cost_scenarios"]["base"]["executed_fills"] for m in first["months"]],
            [m["cost_scenarios"]["base"]["executed_fills"] for m in second["months"]],
        )
        self.assertTrue(first["holdout_untouched"])
        self.assertFalse(first["release_approved"])

    def test_shared_book_does_not_add_six_independent_equity_returns(self):
        r = evaluate_joint(self.all_reports())
        first = r["months"][0]["cost_scenarios"]["base"]
        self.assertEqual(first["executed_fills"], 1)
        self.assertEqual(first["skipped_fills"]["symbol_overlap"], 5)
        self.assertFalse(r["release_approved"])

    def test_joint_priority_never_looks_at_future_duration_or_outcome(self):
        # All six model hypotheses enter same coin/minute. The priority of
        # TARGETS[0] (LONG 5m) is fixed BEFORE future price paths are known.
        # SHORT alternatives exit earlier and have larger hindsight profits.
        from research.execution import net_after_costs
        for long_duration,short_duration in ((6,1),(1,6)):
            reports=self.all_reports()
            for report in reports:
                proposal=report["joint_portfolio_candidates"][0]
                entry=__import__("pandas").Timestamp(proposal["entry_at"])
                is_first=report["target"] == TARGETS[0]
                duration=long_duration if is_first else short_duration
                proposal["exit_at"]=(entry+__import__("pandas").Timedelta(
                    minutes=duration)).isoformat()
                proposal["gross_return"]=-.003 if is_first else .03
            combined=evaluate_joint(reports)
            first=combined["months"][0]["cost_scenarios"]["base"]
            expected_net=-net_after_costs(-.01,6,3)
            # Per executed notional the first negative gross is unchanged.
            expected_return=net_after_costs(-.003,6,3)
            expected_notional=min(10000*.01/expected_net,10000*6,10000*20)
            expected_equity_return=expected_notional*expected_return/10000
            self.assertEqual(first["executed_fills"],1)
            self.assertAlmostEqual(first["net_equity_return"],
                                   expected_equity_return,places=11)
            self.assertEqual(combined["simultaneous_same_symbol_signal_priority"],TARGETS)
            self.assertFalse(combined["future_exit_timestamps_used_for_admission_priority"])
            self.assertFalse(combined["release_approved"])

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

    def test_forged_candidate_in_locked_holdout_fails_closed(self):
        reports=self.all_reports()
        proposal=reports[0]["joint_portfolio_candidates"][0]
        proposal["entry_at"]="2026-08-17T00:00:00Z"
        proposal["exit_at"]="2026-08-17T00:05:00Z"
        with self.assertRaisesRegex(ValueError,"candidate crosses"):
            evaluate_joint(reports)

    def test_candidate_outside_own_fold_is_rejected(self):
        from pandas import Timestamp, Timedelta
        for entry in ("2026-04-16T23:59:00Z", "2026-05-17T00:00:00Z"):
            reports=self.all_reports()
            proposal=reports[0]["joint_portfolio_candidates"][0]
            proposal["entry_at"]=entry
            proposal["exit_at"]=(Timestamp(entry)+Timedelta(minutes=5)).isoformat()
            with self.assertRaisesRegex(ValueError,"candidate crosses"):
                evaluate_joint(reports)

    def test_candidate_exit_in_holdout_is_rejected_even_with_valid_entry(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["exit_at"]="2026-08-17T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"candidate crosses"):
            evaluate_joint(reports)

    def test_naive_or_reversed_candidate_clock_fails_closed(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["entry_at"]="2026-04-19 00:00:00"
        with self.assertRaisesRegex(ValueError,"timezone-aware"):
            evaluate_joint(reports)
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["exit_at"]="2026-04-18T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"candidate crosses"):
            evaluate_joint(reports)

    def test_fold_end_metadata_must_match_frozen_development_period(self):
        for broken in ("2026-08-17T00:00:00Z", None):
            reports=self.all_reports()
            if broken is None:
                del reports[0]["test_end_exclusive"]
            else:
                reports[0]["test_end_exclusive"]=broken
            with self.assertRaisesRegex(ValueError,"development fold"):
                evaluate_joint(reports)

    def test_late_preholdout_exit_is_counted_not_hidden_in_monthly_claim(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["exit_at"]="2026-05-18T00:00:00Z"
        result=evaluate_joint(reports)
        self.assertEqual(result["months"][0]["candidate_exits_at_or_after_fold_end"],1)
        self.assertTrue(result["fold_entry_and_locked_holdout_windows_validated"])
        self.assertFalse(result["release_approved"])

    def test_utc_folds_accept_negative_offset_on_previous_local_day(self):
        reports=self.all_reports()
        reports[0]["test_start"]="2026-04-16T21:00:00-03:00"
        reports[0]["test_end_exclusive"]="2026-05-16T21:00:00-03:00"
        reports[0]["holdout_locked_from"]="2026-08-16T21:00:00-03:00"
        result=evaluate_joint(reports)
        self.assertTrue(result["fold_entry_and_locked_holdout_windows_validated"])

    def test_accepted_late_exit_invalidates_only_as_of_month_goal(self):
        reports=self.all_reports()
        # Target[0] has the fixed predeclared priority at entry.
        reports[0]["joint_portfolio_candidates"][0]["exit_at"]="2026-05-18T00:00:00Z"
        result=evaluate_joint(reports)
        april=result["months"][0]
        self.assertEqual(april["accepted_base_positions_settling_after_fold"],1)
        self.assertFalse(april["base_cost_30d_goal_assessable"])
        self.assertIsNone(april["base_cost_net_return_at_fold_end"])
        self.assertIsNone(april["net_7pct_at_base_cost"])
        self.assertFalse(result["all_four_dev_months_30d_goal_assessable"])
        self.assertFalse(result["all_four_dev_months_net7"])
        self.assertIsNotNone(april["cost_scenarios"]["base"]["net_equity_return"])
        self.assertIsNone(april["cost_scenarios"]["base"]["accepted_position_trace"])
        self.assertFalse(result["release_approved"])

    def test_late_rejected_candidate_does_not_invalidate_month(self):
        reports=self.all_reports()
        # TARGETS[1] enters at exactly the same minute, but TARGETS[0]
        # already takes the sole BTC slot. Future outcome of rejected trade
        # must not affect month completeness.
        reports[1]["joint_portfolio_candidates"][0]["exit_at"]="2026-05-18T00:00:00Z"
        result=evaluate_joint(reports)
        april=result["months"][0]
        self.assertEqual(april["candidate_exits_at_or_after_fold_end"],1)
        self.assertEqual(april["accepted_base_positions_settling_after_fold"],0)
        self.assertTrue(april["base_cost_30d_goal_assessable"])
        self.assertIsNotNone(april["base_cost_net_return_at_fold_end"])
        self.assertIsNotNone(april["net_7pct_at_base_cost"])

    def test_every_cost_scenario_hides_unobservable_30day_goal(self):
        reports=self.all_reports()
        proposal=reports[0]["joint_portfolio_candidates"][0]
        proposal["exit_at"]="2026-05-18T00:00:00Z"
        # A large profit only realized AFTER month end must not manufacture
        # a positive 30-day goal decision from hindsight cohort PnL.
        proposal["gross_return"]=.20
        result=evaluate_joint(reports)
        april=result["months"][0]
        for name, scenario in april["cost_scenarios"].items():
            with self.subTest(scenario=name):
                self.assertEqual(scenario["accepted_positions_settling_after_fold_end"],1)
                self.assertFalse(scenario["net_30d_goal_assessable"])
                self.assertIsNone(scenario["net_30d_return_at_fold_end"])
                self.assertIsNone(scenario["net_30d_goal_met_in_development"])
                self.assertTrue(scenario["full_entry_cohort_goal_met_after_final_settlement"])
                self.assertIsNone(scenario["accepted_position_trace"])

    def test_uncontaminated_month_remains_assessable_in_every_scenario(self):
        report=evaluate_joint(self.all_reports())
        april=report["months"][0]
        self.assertTrue(report["all_four_dev_months_30d_goal_assessable"])
        for scenario in april["cost_scenarios"].values():
            self.assertTrue(scenario["net_30d_goal_assessable"])
            self.assertEqual(scenario["accepted_positions_settling_after_fold_end"],0)
            self.assertIsNotNone(scenario["net_30d_return_at_fold_end"])
            self.assertIsInstance(scenario["net_30d_goal_met_in_development"],bool)

    def test_funding_replay_independently_requires_month_end_settlement(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["exit_at"]="2026-05-18T00:00:00Z"
        funded=evaluate_joint(
            reports,
            funding_resolver=lambda fill: replace(
                fill, funding_fraction=-.001, funding_reconciled=True,
            ),
        )
        april=funded["months"][0]
        self.assertEqual(april["accepted_funding_positions_settling_after_fold"],1)
        self.assertFalse(april["funding_adjusted_30d_goal_assessable"])
        self.assertIsNone(april["funding_adjusted_net_return_at_fold_end"])
        self.assertIsNone(april["net_7pct_at_base_cost_after_funding"])
        self.assertFalse(funded["all_four_dev_months_net7_after_archived_funding"])
        for scenario in april["funding_adjusted_cost_scenarios"].values():
            self.assertEqual(scenario["accepted_positions_settling_after_fold_end"],1)
            self.assertFalse(scenario["net_30d_goal_assessable"])
            self.assertIsNone(scenario["net_30d_goal_met_in_development"])
            self.assertIsNone(scenario["accepted_position_trace"])
        self.assertFalse(funded["release_approved"])

    def test_same_utc_boundaries_allow_explicit_timezone_offset(self):
        reports=self.all_reports()
        reports[0]["test_start"]="2026-04-17T03:00:00+03:00"
        reports[0]["test_end_exclusive"]="2026-05-17T03:00:00+03:00"
        result=evaluate_joint(reports)
        self.assertTrue(result["holdout_untouched"])

    def test_wrong_direction_rejected(self):
        reports=self.all_reports()
        reports[0]["joint_portfolio_candidates"][0]["side"]="short"
        with self.assertRaises(ValueError): evaluate_joint(reports)

    def test_gate_count_mismatch_fails(self):
        reports=self.all_reports()
        reports[0]["research_abstention"]["gate_admitted"]=2
        with self.assertRaises(ValueError): evaluate_joint(reports)

if __name__=="__main__": unittest.main()
