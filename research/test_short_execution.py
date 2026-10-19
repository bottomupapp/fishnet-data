import unittest

import numpy as np
import pandas as pd

from research.execution import net_after_costs_short, simulate_short


class ShortExecutionTests(unittest.TestCase):
    def setUp(self):
        self.t = pd.date_range("2026-07-20", periods=90, freq="min", tz="UTC")
        self.bars = pd.DataFrame({"open": np.full(90, 100.),
                                  "high": np.full(90, 100.1),
                                  "low": np.full(90, 99.9),
                                  "close": np.full(90, 100.)}, index=self.t)

    def test_profitable_short_entry_and_cost_deduction(self):
        b = self.bars.copy()
        b.loc[self.t[2], "low"] = 99.4
        result = simulate_short(b, self.t[0], .002, 15)
        self.assertEqual(result.entry_at, self.t[2].isoformat())
        self.assertEqual(result.exit_reason, "target")
        self.assertAlmostEqual(result.gross_return, .002)
        self.assertLess(result.net_return, result.gross_return)

    def test_short_stop_has_negative_net_and_stop_first(self):
        b = self.bars.copy()
        b.loc[self.t[2], ["high", "low"]] = [100.5, 99.5]
        result = simulate_short(b, self.t[0], .002, 15)
        self.assertEqual(result.exit_reason, "stop")
        self.assertLess(result.net_return, -.002)

    def test_short_gap_through_stop_uses_open(self):
        b = self.bars.copy()
        b.loc[self.t[3], ["open", "high", "low", "close"]] = [101, 101.2, 100.9, 101.1]
        result = simulate_short(b, self.t[0], .002, 15)
        self.assertEqual(result.exit_reason, "stop_gap")
        self.assertAlmostEqual(result.gross_return, -.01)

    def test_missing_minute_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing minute"):
            simulate_short(self.bars.drop(self.t[3]), self.t[0], .002, 15)

    def test_no_fill_without_sufficient_future_window(self):
        with self.assertRaises(ValueError):
            simulate_short(self.bars, self.t[80], .002, 15)

    def test_short_flat_return_negative_after_costs(self):
        self.assertLess(net_after_costs_short(0., 6., 3.), 0)
        self.assertGreater(net_after_costs_short(.01, 6., 3.), 0)

    def test_short_net_return_worsens_as_costs_rise(self):
        gross=.004
        self.assertGreater(net_after_costs_short(gross, 3, 1), net_after_costs_short(gross, 10, 6))

    def test_bad_short_costs_and_return_rejected(self):
        for gr, fee, slip in ((float("nan"), 6, 3), (.01, -1, 3), (1.01, 6, 3)):
            with self.assertRaises(ValueError):
                net_after_costs_short(gr, fee, slip)

if __name__ == "__main__":
    unittest.main()
