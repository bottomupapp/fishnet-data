import unittest

import numpy as np
import pandas as pd

from research.execution import simulate_long, net_after_costs


class ConservativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2026-07-20", periods=12, freq="min", tz="UTC")
        self.bars = pd.DataFrame(
            {"open": np.full(12, 100.), "high": np.full(12, 100.1),
             "low": np.full(12, 99.9), "close": np.full(12, 100.)},
            index=self.times,
        )

    def test_first_fill_is_t_plus_two_open(self):
        bars = self.bars.copy()
        bars.loc[self.times[1], ["open", "high", "low", "close"]] = [110, 110.1, 109.9, 110]
        result = simulate_long(bars, self.times[0], .002, 4)
        self.assertEqual(result.entry_at, self.times[2].isoformat())
        self.assertEqual(result.exit_reason, "time_limit")
        self.assertLess(result.net_return, result.gross_return)

    def test_target_deducts_fee_and_slippage(self):
        bars = self.bars.copy()
        bars.loc[self.times[2], "high"] = 100.4
        result = simulate_long(bars, self.times[0], .002, 5)
        self.assertEqual(result.exit_reason, "target")
        self.assertAlmostEqual(result.gross_return, .002, places=8)
        self.assertGreater(result.net_return, 0)
        self.assertLess(result.net_return, .002)

    def test_same_bar_target_and_stop_is_loss(self):
        bars = self.bars.copy()
        bars.loc[self.times[2], ["high", "low"]] = [100.5, 99.5]
        result = simulate_long(bars, self.times[0], .002, 5)
        self.assertEqual(result.exit_reason, "stop")
        self.assertLess(result.net_return, -.002)

    def test_gap_through_stop_uses_worse_open(self):
        bars = self.bars.copy()
        bars.loc[self.times[3], ["open", "high", "low", "close"]] = [99, 99.2, 98.9, 99.1]
        result = simulate_long(bars, self.times[0], .002, 5)
        self.assertEqual(result.exit_reason, "stop_gap")
        self.assertAlmostEqual(result.gross_return, -.01, places=8)

    def test_gap_in_minute_index_is_rejected(self):
        bars = self.bars.drop(index=self.times[3])
        with self.assertRaisesRegex(ValueError, "missing minute"):
            simulate_long(bars, self.times[0], .002, 5)

    def test_imputed_bar_is_rejected(self):
        bars = self.bars.copy()
        bars["source_observed"] = True
        bars.loc[self.times[3], "source_observed"] = False
        with self.assertRaisesRegex(ValueError, "synthetic"):
            simulate_long(bars, self.times[0], .002, 5)

    def test_needs_complete_future_horizon(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            simulate_long(self.bars, self.times[8], .002, 5)

    def test_higher_costs_cannot_improve_return(self):
        gross = 0.003
        lower = net_after_costs(gross, 3, 1)
        base = net_after_costs(gross, 6, 3)
        adverse = net_after_costs(gross, 10, 6)
        self.assertGreater(lower, base)
        self.assertGreater(base, adverse)

    def test_cost_utility_rejects_invalid_assumptions(self):
        for gross, fee, slip in ((float("nan"), 6, 3), (0.001, -1, 3), (-1.1, 6, 3)):
            with self.assertRaises(ValueError):
                net_after_costs(gross, fee, slip)

    def test_cannot_assume_impossible_immediate_fill(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            simulate_long(self.bars, self.times[0], .002, 5, entry_delay_minutes=1)


if __name__ == "__main__":
    unittest.main()
