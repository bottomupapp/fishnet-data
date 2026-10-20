import unittest
import numpy as np
import pandas as pd

from research.rr_labels import REWARD_MULTIPLE, HOLDOUT_START, label_rr_events, rr_price_target
from research.execution import net_after_costs, net_after_costs_short


class Net32RResearchLabels(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2026-07-20", periods=45, freq="min", tz="UTC")
        self.bars = pd.DataFrame({
            "open": np.full(len(self.idx), 100.),
            "high": np.full(len(self.idx), 100.1),
            "low": np.full(len(self.idx), 99.9),
            "close": np.full(len(self.idx), 100.),
            "source_observed": True,
        }, index=self.idx)

    def calc(self, side="long", horizon=15, t=0):
        return label_rr_events(self.bars, pd.DatetimeIndex([self.idx[t]]),
            np.array([.005]), side=side, horizon_minutes=horizon).iloc[0]

    def test_long_target_pays_exact_three_point_two_r_after_costs(self):
        price_move, stop_loss = rr_price_target(.005, "long")
        r = net_after_costs(price_move, 6, 3) / stop_loss
        self.assertAlmostEqual(r, 3.2, places=10)
        self.bars.loc[self.idx[3], "high"] = 100 * (1 + price_move + .0001)
        outcome = self.calc()
        self.assertEqual(outcome.label, 1)
        self.assertEqual(outcome.reason, "target")
        self.assertAlmostEqual(outcome.net_r, 3.2, places=8)
        self.assertEqual(outcome.entry_at, self.idx[2])

    def test_short_target_pays_exact_three_point_two_r_after_costs(self):
        price_move, stop_loss = rr_price_target(.005, "short")
        self.assertAlmostEqual(net_after_costs_short(price_move,6,3)/stop_loss,3.2,places=10)
        self.bars.loc[self.idx[3], "low"] = 100 * (1 - price_move - .0001)
        result = self.calc("short")
        self.assertEqual(result.label, 1)
        self.assertAlmostEqual(result.net_r, 3.2, places=8)

    def test_simultaneous_target_and_stop_is_loss_for_both_sides(self):
        long_move,_ = rr_price_target(.005,"long")
        short_move,_ = rr_price_target(.005,"short")
        self.bars.loc[self.idx[3], "high"] = 100 * (1 + max(.005,long_move) + .001)
        self.bars.loc[self.idx[3], "low"] = 100 * (1 - max(.005,short_move) - .001)
        for side in ("long","short"):
            result = self.calc(side)
            self.assertEqual(result.label, 0)
            self.assertEqual(result.reason,"stop")
            self.assertAlmostEqual(result.net_r,-1,places=8)

    def test_gap_stop_can_exceed_one_r(self):
        self.bars.loc[self.idx[3], ["open","high","low","close"]] = [98, 98.2, 97.9, 98.1]
        result=self.calc("long")
        self.assertEqual(result.reason,"stop_gap")
        self.assertLess(result.net_r,-1)

    def test_time_limit_is_not_positive_label_even_if_unrealized_return_positive(self):
        self.bars.loc[self.idx[16], "close"] = 100.2
        self.bars.loc[self.idx[16], "high"] = 100.2
        result=self.calc()
        self.assertEqual(result.label,0)
        self.assertEqual(result.reason,"time_limit")

    def test_missing_minute_fails_closed(self):
        self.bars.loc[self.idx[5], "source_observed"] = False
        result=self.calc()
        self.assertTrue(np.isnan(result.label))
        self.assertEqual(result.reason,"invalid")

    def test_missing_timestamp_fails_closed(self):
        self.bars=self.bars.drop(index=self.idx[5])
        result=self.calc()
        self.assertTrue(np.isnan(result.label))

    def test_holdout_crossing_is_invalid(self):
        dates = pd.date_range(HOLDOUT_START-pd.Timedelta(minutes=20),periods=len(self.bars),freq="min")
        bars=self.bars.copy()
        bars.index=dates
        result=label_rr_events(bars,pd.DatetimeIndex([dates[10]]),
            np.array([.005]),side="long",horizon_minutes=15).iloc[0]
        self.assertTrue(np.isnan(result.label))

    def test_decision_at_t_plus_one_would_be_optimistic(self):
        self.bars.loc[self.idx[1], ["open","high","low","close"]] = [100, 120, 99.9, 100]
        result=self.calc()
        self.assertEqual(result.entry_at,self.idx[2])
        self.assertEqual(result.label,0)

    def test_invalid_direction_and_geometry_fail_closed(self):
        with self.assertRaises(ValueError): rr_price_target(.005,"buy")
        with self.assertRaises(ValueError): rr_price_target(0,"long")
        with self.assertRaises(ValueError): rr_price_target(.005,"short",reward_multiple=-1)
        with self.assertRaises(ValueError): rr_price_target(.5,"short",reward_multiple=10.)

    def test_fees_raise_gross_price_target(self):
        cheap=rr_price_target(.005,"long",fee_bps_per_side=3,slippage_bps_per_side=1)[0]
        expensive=rr_price_target(.005,"long",fee_bps_per_side=10,slippage_bps_per_side=6)[0]
        self.assertGreater(expensive,cheap)

    def test_shape_mismatch_and_duplicate_decisions_rejected(self):
        with self.assertRaises(ValueError):
            label_rr_events(self.bars,pd.DatetimeIndex([self.idx[0]]),np.array([]),side="long")
        with self.assertRaises(ValueError):
            label_rr_events(self.bars,pd.DatetimeIndex([self.idx[0],self.idx[0]]),
                            np.array([.005,.005]),side="long")

if __name__=="__main__": unittest.main()
