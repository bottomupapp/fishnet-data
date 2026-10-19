import unittest

from research.grid_summary import TARGETS, FOLD_STARTS, aggregate


def a_report(target, value=.08, fills=25):
    return {
        "target": target,
        "release_approved": False,
        "holdout_start": "2026-08-17T00:00:00Z",
        "folds": [{
            "test_start": day,
            "risk_sized_portfolio_30d": {
                "abstention": {
                    "base": {
                        "net_equity_return": value,
                        "executed_fills": fills,
                        "max_realized_only_drawdown": .01,
                        "mean_realized_r": .4,
                        "target_met": value >= .07,
                    },
                },
            },
        } for day in FOLD_STARTS],
    }


class GridSummaryTests(unittest.TestCase):
    def test_all_six_scenarios_are_required_and_never_released(self):
        result=aggregate([a_report(t) for t in TARGETS])
        self.assertEqual(result["evaluated_distinct_targets"],6)
        self.assertFalse(result["release_approved"])
        self.assertTrue(all(x["research_consistency_passed"] for x in result["targets"]))
        self.assertTrue(all(not x["release_approved"] for x in result["targets"]))

    def test_one_negative_development_month_blocks_target(self):
        reports=[a_report(t) for t in TARGETS]
        reports[1]["folds"][0]["risk_sized_portfolio_30d"]["abstention"]["base"]["net_equity_return"] = -.01
        reports[1]["folds"][0]["risk_sized_portfolio_30d"]["abstention"]["base"]["target_met"] = False
        result=aggregate(reports)
        self.assertFalse(result["targets"][1]["research_consistency_passed"])

    def test_missing_or_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([a_report(t) for t in TARGETS[:-1]])
        reports=[a_report(t) for t in TARGETS]
        reports[3]["target"]=reports[0]["target"]
        with self.assertRaises(ValueError):
            aggregate(reports)

    def test_holdout_change_rejected(self):
        reports=[a_report(t) for t in TARGETS]
        reports[0]["holdout_start"]="2026-08-01T00:00:00Z"
        with self.assertRaises(ValueError):
            aggregate(reports)

    def test_unapproved_artifacts_only(self):
        reports=[a_report(t) for t in TARGETS]
        reports[0]["release_approved"]=True
        with self.assertRaises(ValueError):
            aggregate(reports)

    def test_insufficient_monthly_trades_prevents_consistency(self):
        result=aggregate([a_report(t,fills=3) for t in TARGETS])
        self.assertTrue(all(not x["research_consistency_passed"] for x in result["targets"]))

if __name__=="__main__":
    unittest.main()
