import unittest
import numpy as np
import pandas as pd

from research.regime_gate import (
    REGIMES, ACTIVITY_Z_FLOOR, regime_of,
    fit_regime_gates, select_from_frozen_gates,
)


def calibration_fixture(net_r=1.0):
    n=2400
    dates=pd.date_range("2026-06-01",periods=n,freq="5min",tz="UTC")
    k=np.arange(n)%4
    return pd.DataFrame({
        "t":dates,
        "symbol":np.where(np.arange(n)%2==0,"BTCUSDT","ETHUSDT"),
        "ret_240":np.where(np.isin(k,(0,1)), .005, -.005),
        "rv_60":np.where(np.isin(k,(1,3)),1.4,.75),
        "vol15_z":np.zeros(n),
        "net_r":np.full(n,net_r),
    })


class RegimeGateSafetyTests(unittest.TestCase):
    def test_regimes_are_ex_ante_and_direction_sensitive(self):
        data=calibration_fixture()
        a=regime_of(data,side="long")
        b=regime_of(data,side="short")
        self.assertEqual(set(a),set(REGIMES))
        self.assertEqual(set(b),set(REGIMES))
        self.assertEqual(a.iloc[0],"aligned_calm")
        self.assertEqual(b.iloc[0],"counter_calm")

    def test_missing_or_illiquid_activity_fails_closed(self):
        data=calibration_fixture().iloc[:4].copy()
        data.loc[data.index[0],"vol15_z"]=ACTIVITY_Z_FLOOR-0.1
        data.loc[data.index[1],"rv_60"]=float("nan")
        self.assertEqual(regime_of(data,side="long").iloc[0],"excluded")
        self.assertEqual(regime_of(data,side="long").iloc[1],"excluded")
        with self.assertRaises(ValueError):
            regime_of(data.drop(columns=["rv_60"]),side="long")

    def test_negative_calibration_net_r_denies_all_test_trades(self):
        cal=calibration_fixture(net_r=-.5)
        scores=np.linspace(-.1,.9,len(cal))
        gates=fit_regime_gates(cal,scores,side="long",horizon_minutes=60)
        self.assertEqual(set(gates),set(REGIMES))
        self.assertTrue(all(not g.eligible_for_research for g in gates.values()))
        test=cal.drop(columns=["net_r"]).copy()
        self.assertTrue(select_from_frozen_gates(
            test,np.ones(len(test)),gates,side="long",horizon_minutes=60
        ).empty)

    def test_positive_calibration_allows_candidate_without_test_labels(self):
        cal=calibration_fixture(net_r=1.5)
        scores=np.linspace(0,1,len(cal))
        gates=fit_regime_gates(cal,scores,side="long",horizon_minutes=60)
        self.assertTrue(any(g.eligible_for_research for g in gates.values()))
        test=cal.iloc[:200].drop(columns=["net_r"]).copy()
        chosen=select_from_frozen_gates(
            test,np.ones(len(test)),gates,side="long",horizon_minutes=60)
        self.assertGreater(len(chosen),0)
        self.assertTrue(set(chosen.regime).issubset(set(REGIMES)))

    def test_pretest_gate_ignores_future_outcome_columns(self):
        cal=calibration_fixture(net_r=1.5)
        scores=np.linspace(0,1,len(cal))
        gates=fit_regime_gates(cal,scores,side="short",horizon_minutes=60)
        test=cal.iloc[:250].copy()
        chosen1=select_from_frozen_gates(test,np.ones(len(test)),
                                         gates,side="short",horizon_minutes=60)
        test["net_r"]=-9999.
        chosen2=select_from_frozen_gates(test,np.ones(len(test)),
                                         gates,side="short",horizon_minutes=60)
        self.assertEqual(list(chosen1.index),list(chosen2.index))

    def test_insufficient_nonoverlapping_evidence_is_rejected(self):
        cal=calibration_fixture(net_r=1.5)
        scores=np.linspace(0,1,len(cal))
        gates=fit_regime_gates(cal,scores,side="long",horizon_minutes=240)
        self.assertTrue(all(not x.eligible_for_research
                            for x in gates.values()))

    def test_nonfinite_scores_and_calibration_outcomes_rejected(self):
        cal=calibration_fixture()
        with self.assertRaises(ValueError):
            fit_regime_gates(cal,np.full(len(cal),float("nan")),
                             side="long",horizon_minutes=60)
        cal.loc[cal.index[0],"net_r"]=float("nan")
        with self.assertRaises(ValueError):
            fit_regime_gates(cal,np.zeros(len(cal)),
                             side="long",horizon_minutes=60)


if __name__=="__main__":
    unittest.main()
