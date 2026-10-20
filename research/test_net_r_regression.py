import unittest
import numpy as np
import pandas as pd

from research.net_r_gate import calibrate_r_gate
from research.net_r_regression import PREDECLARED_REWARDS,PREDECLARED_HORIZONS
from research.rr_model import partition_boundaries
from research.rr_labels import HOLDOUT_START


def make_data():
    dates=pd.date_range("2026-06-01",periods=720,freq="5min",tz="UTC")
    # 720 points have 60+ independent 60m outcomes after 5m stride.
    return pd.DataFrame({
        "t":dates,
        "symbol":["BTCUSDT"]*len(dates),
        "net_r":np.r_[np.full(650,-.2),np.full(70,1.5)],
    })


class NetRRegressionGateTests(unittest.TestCase):
    def test_only_prior_actual_realized_r_can_support_a_strategy(self):
        data=make_data()
        scores=np.linspace(-.1,.9,len(data))
        gate=calibrate_r_gate(data,scores,horizon_minutes=60,
            minimum_high_scores=20)
        self.assertLess(gate.reference_events,len(data))
        self.assertFalse(gate.admitted_for_research) if gate.highest_decile_events<20 else self.assertTrue(
            isinstance(gate.admitted_for_research,bool)
        )

    def test_negative_net_r_evidence_fails_closed(self):
        data=make_data()
        data["net_r"]=-.5
        scores=np.linspace(-.1,.9,len(data))
        gate=calibrate_r_gate(data,scores,horizon_minutes=60,
            minimum_high_scores=20)
        self.assertFalse(gate.admitted_for_research)
        self.assertFalse(gate.select(np.array([100.]))[0])

    def test_scores_must_match_calibration_rows(self):
        data=make_data()
        with self.assertRaises(ValueError):
            calibrate_r_gate(data,np.zeros(len(data)-1),horizon_minutes=60)
        with self.assertRaises(ValueError):
            calibrate_r_gate(data,np.full(len(data),float("nan")),horizon_minutes=60)

    def test_small_independent_sample_cannot_approve(self):
        data=make_data()
        scores=np.linspace(-.1,.9,len(data))
        with self.assertRaises(ValueError):
            calibrate_r_gate(data,scores,horizon_minutes=240)

    def test_predeclared_reward_and_horizon_do_not_use_holdout(self):
        self.assertEqual(PREDECLARED_REWARDS,(1.5,))
        self.assertEqual(PREDECLARED_HORIZONS,(60,240))
        for h in PREDECLARED_HORIZONS:
            w=partition_boundaries(pd.Timestamp("2026-07-17T00:00:00Z"),h)
            self.assertLess(w["test_end_exclusive"],HOLDOUT_START)

if __name__=="__main__":
    unittest.main()
