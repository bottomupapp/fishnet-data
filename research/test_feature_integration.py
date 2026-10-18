import unittest
import numpy as np
import pandas as pd
from research.features import labels
from research.quality import mask_labels, valid_history

class FeatureIntegrationTests(unittest.TestCase):
    def test_missing_bar_invalidates_context_and_future_outcomes(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC")
        observed = pd.Series(True, index=idx)
        observed.iloc[50] = False
        history = valid_history(observed, 10)
        self.assertFalse(bool(history.iloc[50]))
        self.assertFalse(bool(history.iloc[59]))
        self.assertTrue(bool(history.iloc[60]))
        bars = pd.DataFrame({"open": np.full(100,100.0), "high": np.full(100,100.1),
                             "low": np.full(100,99.9)}, index=idx)
        sigma = pd.Series(np.full(100,0.001), index=idx)
        y = mask_labels(labels(bars,sigma), observed, (5,15,60))
        self.assertTrue(np.isnan(y.iloc[46]["y_long_5"]))
        self.assertTrue(np.isnan(y.iloc[46]["y_short_5"]))
        self.assertFalse(np.isnan(y.iloc[51]["y_long_5"]))

if __name__ == "__main__":
    unittest.main()
