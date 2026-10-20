import unittest
import pandas as pd

from research.rr_model import (
    ECONOMIC_PROBABILITY_FLOOR, PREDECLARED_REWARDS, HOLDOUT_START, partition_boundaries, safe_model_columns
)


class Net32RModelSafetyTests(unittest.TestCase):
    def test_training_calibration_test_outcome_horizons_are_purged(self):
        for horizon in (60,240):
            windows=partition_boundaries(pd.Timestamp("2026-07-17T00:00:00Z"),horizon)
            duration=pd.Timedelta(minutes=horizon+2)
            self.assertLessEqual(windows["train_end_exclusive"]+duration+pd.Timedelta(days=1),
                                 windows["cal_start"])
            self.assertLessEqual(windows["cal_end_exclusive"]+duration+pd.Timedelta(days=1),
                                 windows["test_start"])
            self.assertLessEqual(windows["test_end_exclusive"],HOLDOUT_START-pd.Timedelta(days=1))

    def test_holdout_test_start_rejected(self):
        with self.assertRaises(ValueError):
            partition_boundaries(HOLDOUT_START,240)
        with self.assertRaises(ValueError):
            partition_boundaries(pd.Timestamp("2026-07-17"),60)

    def test_future_targets_cannot_be_used_as_features(self):
        good = pd.DataFrame({
            "t": pd.date_range("2026-07-17", periods=2, freq="5min", tz="UTC"),
            "symbol":["BTCUSDT","ETHUSDT"],
            "sigma_1d":[.01,.01],
            "ret_5":[.001,.002],
            "barrier_60":[.003,.004],
            "y_long_60":[1.,0.]
        })
        self.assertEqual(safe_model_columns(good),["ret_5"])
        bad=good.copy()
        bad["rr_net_r"]=[3.2,-1]
        with self.assertRaises(ValueError):
            safe_model_columns(bad)

    def test_reward_grid_predeclared_and_not_force_fit_to_goal(self):
        self.assertEqual(PREDECLARED_REWARDS, (1.0, 1.5, 2.0, 3.2))
        self.assertAlmostEqual(1/(1+1.),.5)
        self.assertAlmostEqual(1/(1+1.5),.4)
        self.assertAlmostEqual(1/(1+2.),1/3)
        self.assertAlmostEqual(1/(1+3.2),ECONOMIC_PROBABILITY_FLOOR)

    def test_all_reward_options_share_the_same_holdout_protection(self):
        for horizon in (60,240):
            windows=partition_boundaries(pd.Timestamp("2026-07-17T00:00:00Z"),horizon)
            self.assertLessEqual(windows["test_end_exclusive"],HOLDOUT_START-pd.Timedelta(days=1))

    def test_economic_floor_requires_nontrivial_probability(self):
        self.assertAlmostEqual(ECONOMIC_PROBABILITY_FLOOR,1/4.2)
        self.assertGreater(ECONOMIC_PROBABILITY_FLOOR,.23)

if __name__=="__main__":
    unittest.main()
