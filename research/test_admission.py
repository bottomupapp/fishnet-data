import unittest
from research.admission import evaluate_admission, minimum_probability_for_cost_surplus


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

    def test_unreachable_cost_barrier_gives_probability_above_one(self):
        # 20bp barrier vs 18bp roundtrip friction plus 5bp uncertainty:
        # even an impossible 100% barrier hit probability does not pass.
        required = minimum_probability_for_cost_surplus(.002)
        self.assertGreater(required, 1.)
        self.assertFalse(evaluate_admission(1., .002).admitted)

    def test_reachable_break_even_is_exact(self):
        barrier = .01
        required = minimum_probability_for_cost_surplus(barrier)
        self.assertLess(required, 1.)
        self.assertFalse(evaluate_admission(required, barrier).admitted)
        self.assertTrue(evaluate_admission(min(1., required + .001), barrier).admitted)

    def test_fees_make_required_probability_higher(self):
        p1 = minimum_probability_for_cost_surplus(.005, fee_bps_per_side=3, slippage_bps_per_side=1)
        p2 = minimum_probability_for_cost_surplus(.005, fee_bps_per_side=10, slippage_bps_per_side=6)
        self.assertLess(p1,p2)

    def test_invalid_costs_cannot_pass(self):
        with self.assertRaises(ValueError):
            evaluate_admission(.8, .01, fee_bps_per_side=-1)


if __name__ == "__main__":
    unittest.main()
