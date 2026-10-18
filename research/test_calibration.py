import unittest
import numpy as np
import pandas as pd
from research.run import score

class CalibrationTest(unittest.TestCase):
    def test_nonfinite_calibration_rejected(self):
        t = pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC")
        d = pd.DataFrame({"t":t, "symbol":["BTCUSDT"]*2, "y_long_5":[1.,0.]})
        with self.assertRaises(ValueError):
            score(d, np.array([0.8,0.2]), "y_long_5", 5, np.array([np.nan]))
    def test_calibration_quantile_finite(self):
        t = pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC")
        d = pd.DataFrame({"t":t, "symbol":["BTCUSDT"]*2, "y_long_5":[1.,0.]})
        output = score(d, np.array([0.8,0.2]), "y_long_5", 5, np.array([0.5,np.nan,0.6]))
        self.assertTrue(np.isfinite(output["top0.001"]["thr"]))

if __name__ == "__main__":
    unittest.main()
