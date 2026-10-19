import unittest
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
