"""Fail-closed checks for Fishnet research admission diagnostics."""
import unittest
from research.admission import evaluate_admission

class DiagnosticsTests(unittest.TestCase):
    def test_no_future_outcomes_are_required(self):
        cases = [(.9,.003),(.7,.002),(.99,.01)]
        outcomes = [evaluate_admission(p,b) for p,b in cases]
        self.assertEqual(len(outcomes),3)
        self.assertTrue(all(v.margin_after_cost is not None for v in outcomes))
        self.assertEqual(sum(v.admitted for v in outcomes),
                         sum(v.margin_after_cost > 0 for v in outcomes))

    def test_cost_sensitivity_is_monotonic_without_target_tuning(self):
        p,b=.9,.005
        lower=evaluate_admission(p,b,fee_bps_per_side=3,slippage_bps_per_side=1)
        higher=evaluate_admission(p,b,fee_bps_per_side=10,slippage_bps_per_side=6)
        self.assertGreater(lower.margin_after_cost,higher.margin_after_cost)

if __name__=="__main__": unittest.main()
