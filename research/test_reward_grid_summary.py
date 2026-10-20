import unittest

from research.reward_grid_summary import REWARDS,HORIZONS,SIDES,summarize


def report(side,horizon,reward,net=0.,gated=0):
    return {
        "side":side,"horizon_minutes":horizon,"reward_multiple_cost_net":reward,
        "test_start":"2026-07-17T00:00:00+00:00",
        "locked_holdout_start":"2026-08-17T00:00:00+00:00",
        "net_30d_portfolio_goal":.07,
        "gate_uses_calibration_probability":True,
        "economically_gated_candidate_count":gated,
        "release_approved":False,
        "portfolio":{
            "release_approved":False,"risk_fraction_per_trade":.01,
            "net_equity_return":net,
            "net_30d_goal_met_in_development":net>=.07,
            "executed_fills":gated,
            "max_realized_only_drawdown":0.,
        }
    }

def all_reports():
    return [report(s,h,r) for s in SIDES for h in HORIZONS for r in REWARDS]


class RewardGridValidationTests(unittest.TestCase):
    def test_all_twelve_models_fail_closed(self):
        result=summarize(all_reports())
        self.assertEqual(result["number_of_compared_configurations"],12)
        self.assertTrue(result["comparison_only_not_one_portfolio"])
        self.assertFalse(result["release_approved"])
        self.assertEqual(len(result["models"]),12)

    def test_missing_or_duplicate_experiment_rejected(self):
        data=all_reports()
        with self.assertRaises(ValueError):summarize(data[:-1])
        data[-1]=data[0]
        with self.assertRaises(ValueError):summarize(data)

    def test_achieving_one_month_target_is_not_release(self):
        data=all_reports()
        data[0]=report(SIDES[0],HORIZONS[0],REWARDS[0],net=.12,gated=30)
        result=summarize(data)
        self.assertEqual(sum(m["net_7pct_goal_met_in_one_development_window"]
                             for m in result["models"]),1)
        self.assertFalse(result["release_approved"])

    def test_unapproved_or_unvalidated_calibration_rejected(self):
        data=all_reports()
        data[0]["gate_uses_calibration_probability"]=False
        with self.assertRaises(ValueError):summarize(data)
        data=all_reports()
        data[0]["release_approved"]=True
        with self.assertRaises(ValueError):summarize(data)

    def test_holdout_change_and_risk_cap_rejected(self):
        data=all_reports()
        data[0]["locked_holdout_start"]="2026-08-01T00:00:00Z"
        with self.assertRaises(ValueError):summarize(data)
        data=all_reports()
        data[0]["portfolio"]["risk_fraction_per_trade"]=.02
        with self.assertRaises(ValueError):summarize(data)

    def test_inconsistent_seven_percent_flag_rejected(self):
        data=all_reports()
        data[0]["portfolio"]["net_30d_goal_met_in_development"]=True
        with self.assertRaises(ValueError):summarize(data)

if __name__=="__main__":
    unittest.main()
