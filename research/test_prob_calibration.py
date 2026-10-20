import unittest
import numpy as np

from research.prob_calibration import fit_monotone_calibrator
from research.rr_model import ECONOMIC_PROBABILITY_FLOOR


class RareEventCalibrationSafety(unittest.TestCase):
    def test_extreme_raw_prediction_does_not_bypass_rare_event_base_rate(self):
        raw=np.linspace(.001,.99,1000)
        outcomes=np.zeros(1000,dtype=float)
        outcomes[-10:]=1
        cal=fit_monotone_calibrator(raw,outcomes)
        result=cal.predict(np.array([.001,.5,.99]))
        self.assertTrue(np.all(np.diff(result)>=0))
        self.assertLess(float(result[-1]),ECONOMIC_PROBABILITY_FLOOR)
        self.assertEqual(cal.calibration_positives,10)

    def test_monotonicity_even_when_empirical_bins_fluctuate(self):
        raw=np.linspace(.01,.99,2000)
        outcomes=np.zeros(2000,dtype=float)
        outcomes[200:235]=1
        outcomes[600:610]=1
        cal=fit_monotone_calibrator(raw,outcomes,max_bins=10)
        values=cal.predict(np.linspace(0,1,100))
        self.assertTrue(np.all(np.diff(values)>=-1e-15))

    def test_reject_non_binary_outcomes_and_bad_predictions(self):
        raw=np.linspace(.001,.99,400)
        labels=np.r_[np.ones(10),np.zeros(390)]
        with self.assertRaises(ValueError):
            fit_monotone_calibrator(raw,np.r_[np.full(10,.5),np.zeros(390)])
        with self.assertRaises(ValueError):
            fit_monotone_calibrator(np.full(400,np.nan),labels)
        cal=fit_monotone_calibrator(raw,labels)
        with self.assertRaises(ValueError):
            cal.predict(np.array([float("nan")]))
        with self.assertRaises(ValueError):
            cal.predict(np.array([1.2]))

    def test_too_few_hits_fails_closed(self):
        raw=np.linspace(.001,.99,400)
        with self.assertRaises(ValueError):
            fit_monotone_calibrator(raw,np.zeros(400))
        with self.assertRaises(ValueError):
            fit_monotone_calibrator(raw,np.r_[np.ones(4),np.zeros(396)])

    def test_repeated_raw_probabilities_do_not_break_calibration(self):
        raw=np.r_[np.full(500,.01),np.full(500,.09)]
        y=np.r_[np.zeros(490),np.ones(10),np.zeros(475),np.ones(25)]
        cal=fit_monotone_calibrator(raw,y)
        p=cal.predict(np.array([.01,.09]))
        self.assertEqual(len(p),2)
        self.assertTrue(np.all((p>=0)&(p<=1)))
        self.assertLessEqual(p[0],p[1])

if __name__=="__main__":
    unittest.main()
