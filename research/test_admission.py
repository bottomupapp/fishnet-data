import unittest
from research.admission import evaluate_admission


class AdmissionSafetyTests(unittest.TestCase):
    def test_very_confident_and_wide_barrier_can_pass_research_gate(self):
        outcome = evaluate_admission(.95, .01)
        self.assertTrue(outcome.admitted)
        self.assertEqual(outcome.reason, "research_candidate_only")

    def test_high_hit_probability_can_still_fail_on_narrow_move(self):
        outcome = evaluate_admission(.90, .001)
        self.assertFalse(outcome.admitted)
        self.assertEqual(outcome.reason, "estimated_edge_below_cost")

    def test_higher_fees_never_improve_margin(self):
        low = evaluate_admission(.85, .006, fee_bps_per_side=3, slippage_bps_per_side=1)
        high = evaluate_admission(.85, .006, fee_bps_per_side=10, slippage_bps_per_side=6)
        self.assertGreater(low.margin_after_cost, high.margin_after_cost)

    def test_wider_uncertainty_buffer_never_improves_margin(self):
        low = evaluate_admission(.8, .008, uncertainty_buffer_bps=0)
        high = evaluate_admission(.8, .008, uncertainty_buffer_bps=15)
        self.assertGreater(low.margin_after_cost, high.margin_after_cost)

    def test_nan_and_outside_unit_interval_fail_closed(self):
        for v in (float("nan"), float("inf"), -0.01, 1.01):
            self.assertEqual(evaluate_admission(v, .003).reason, "invalid_probability")

    def test_invalid_barrier_fails_closed(self):
        for v in (0, -0.2, float("nan"), float("inf"), 1):
            self.assertEqual(evaluate_admission(.8, v).reason, "invalid_barrier")

    def test_invalid_costs_cannot_pass(self):
        with self.assertRaises(ValueError):
            evaluate_admission(.8, .01, fee_bps_per_side=-1)


if __name__ == "__main__":
    unittest.main()
