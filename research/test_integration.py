import unittest
import numpy as np
import pandas as pd
from research.features import HORIZONS, labels
from research.quality import mask_labels, valid_history

class IntegrationTest(unittest.TestCase):
    def test_gap_unsafe_labels_and_lookback(self):
        idx = pd.date_range("2026-01-01", periods=1600, freq="min", tz="UTC")
        observed = pd.Series(True, index=idx)
        observed.iloc[1500] = False
        bars = pd.DataFrame({"open": np.full(len(idx), 100.0), "high": np.full(len(idx), 101.0), "low": np.full(len(idx), 99.0)}, index=idx)
        sigma = pd.Series(0.0001, index=idx)
        result = mask_labels(labels(bars, sigma), observed, HORIZONS)
        self.assertTrue(pd.isna(result.iloc[1499]["y_long_5"]))
        self.assertTrue(pd.isna(result.iloc[1490]["y_short_15"]))
        self.assertTrue(pd.isna(result.iloc[1450]["y_long_60"]))
        self.assertFalse(pd.isna(result.iloc[1400]["y_long_60"]))
        self.assertFalse(valid_history(observed, 1440).iloc[1501])
        self.assertTrue(valid_history(observed, 1440).iloc[1499])

if __name__ == "__main__":
    unittest.main()
