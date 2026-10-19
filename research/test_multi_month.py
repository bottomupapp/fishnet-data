import unittest

from research.multi_month import FOLDS, SCENARIOS, summarize


def report(start: str, mean: float = .0001, selected: int = 25) -> dict:
    return {
        "target": "y_long_15",
        "test_start": f"{start} 00:00:00+00:00",
        "test_end_exclusive": "2026-08-16 00:00:00+00:00",
        "rows": {"test": 8640},
        "risk_sized_portfolio_30d": {
            "release_approved": False,
            "risk_fraction_per_planned_stop": .01,
            "target_net_return_per_30_day_dev_window": .07,
            "ungated_cost_sensitivity": {
                name: {"release_approved": False, "risk_fraction_per_trade": .01,
                       "net_equity_return": mean, "net_30d_goal_met_in_development": mean >= .07,
                       "executed_fills": selected, "max_realized_only_drawdown": 0,
                       "max_active_gross_exposure_to_equity": 1,
                       "highest_effective_leverage_used": 5,
                       "mean_realized_r_on_planned_stop_risk": .5}
                for name in SCENARIOS
            },
            "abstention_cost_sensitivity": {
                name: {"release_approved": False, "risk_fraction_per_trade": .01,
                       "net_equity_return": mean, "net_30d_goal_met_in_development": mean >= .07,
                       "executed_fills": selected, "max_realized_only_drawdown": 0,
                       "max_active_gross_exposure_to_equity": 1,
                       "highest_effective_leverage_used": 5,
                       "mean_realized_r_on_planned_stop_risk": .5}
                for name in SCENARIOS
            },
        },
        "portfolio_30d": {
            "release_approved": False,
            "ungated_cost_sensitivity": {
                name: {"release_approved": False, "target_net_30d_return": .07,
                       "net_equity_return": mean, "max_realized_drawdown": 0.0,
                       "executed_fills": selected,
                       "net_30d_goal_met_in_development": mean >= .07}
                for name in SCENARIOS
            },
            "abstention_cost_sensitivity": {
                name: {"release_approved": False, "target_net_30d_return": .07,
                       "net_equity_return": mean, "max_realized_drawdown": 0.0,
                       "executed_fills": selected,
                       "net_30d_goal_met_in_development": mean >= .07}
                for name in SCENARIOS
            },
        },
        "research_abstention": {
            "release_approved": False,
            "gate_admitted": selected,
            "admitted_simulated": selected,
            "mean_net_return_per_admitted_trade": mean,
            "gate_decisions": {"research_candidate_only": selected},
        },
        "execution_audit": {
            "status": "illustrative_cost_assumptions_not_live_execution",
            "selected": selected,
            "simulated": selected,
            "cost_sensitivity": [
                {"scenario": name, "mean_net_return_per_simulated_trade": mean,
                 "net_positive_fraction": .60} for name in SCENARIOS
            ],
        },
    }


class MultiMonthReleaseSafetyTests(unittest.TestCase):
    def test_never_auto_approves_even_if_all_folds_positive(self):
        summary = summarize([report(x) for x in FOLDS])
        self.assertTrue(summary["development_consistency_gate_passed"])
        self.assertFalse(summary["release_approved"])

    def test_one_negative_month_fails_development_gate(self):
        reports = [report(x) for x in FOLDS]
        reports[2] = report(FOLDS[2], mean=-.001)
        self.assertFalse(summarize(reports)["development_consistency_gate_passed"])

    def test_small_signal_sample_fails_development_gate(self):
        reports = [report(x) for x in FOLDS]
        reports[0] = report(FOLDS[0], selected=12)
        self.assertFalse(summarize(reports)["development_consistency_gate_passed"])

    def test_gate_cannot_approve_negative_month(self):
        reports = [report(x) for x in FOLDS]
        reports[1]["research_abstention"]["mean_net_return_per_admitted_trade"] = -.0005
        result = summarize(reports)
        self.assertFalse(result["abstention_development_consistency_gate_passed"])
        self.assertFalse(result["release_approved"])

    def test_gate_refuses_invalid_count(self):
        reports = [report(x) for x in FOLDS]
        reports[1]["research_abstention"]["gate_admitted"] = 26
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_not_seven_percent_though_positive_returns(self):
        data = summarize([report(x, mean=.001) for x in FOLDS])
        self.assertFalse(data["net_7pct_30d_development_goal_met_all_folds"])
        self.assertFalse(data["release_approved"])

    def test_development_goal_cannot_promote_model(self):
        data = summarize([report(x, mean=.08) for x in FOLDS])
        self.assertTrue(data["net_7pct_30d_development_goal_met_all_folds"])
        self.assertFalse(data["release_approved"])

    def test_portfolio_not_approved_fails_closed(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["portfolio_30d"]["release_approved"] = True
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_risk_sized_net_seven_percent_goal_remains_research_only(self):
        reports = [report(x, mean=.08) for x in FOLDS]
        result = summarize(reports)
        self.assertTrue(result["risk_sized_net_7pct_30d_development_goal_met_all_folds"])
        self.assertFalse(result["release_approved"])

    def test_invalid_risk_fraction_fails_closed(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["risk_sized_portfolio_30d"]["risk_fraction_per_planned_stop"] = .02
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_short_and_long_models_must_not_mix_across_folds(self):
        reports = [report(x) for x in FOLDS]
        reports[2]["target"] = "y_short_60"
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_short_horizon_is_preserved(self):
        reports = [report(x) for x in FOLDS]
        for r in reports:
            r["target"] = "y_short_5"
        self.assertEqual(summarize(reports)["target"], "y_short_5")

    def test_missing_month_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([report(x) for x in FOLDS[:3]])

    def test_missing_cost_scenario_is_rejected(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["execution_audit"]["cost_sensitivity"].pop()
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_mismatched_simulation_count_is_rejected(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["execution_audit"]["simulated"] -= 1
        with self.assertRaises(ValueError):
            summarize(reports)


if __name__ == "__main__":
    unittest.main()
