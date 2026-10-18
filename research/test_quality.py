import unittest
import pandas as pd
from research.quality import observed_mask, valid_history, valid_outcome, mask_labels

class QualityTest(unittest.TestCase):
    def setUp(self):
        self.full = pd.date_range("2026-01-01", periods=12, freq="min", tz="UTC")
        self.observed = observed_mask(self.full.delete(5), self.full)

    def test_gap_not_observed(self):
        self.assertFalse(self.observed.iloc[5])
        self.assertTrue(self.observed.iloc[6])

    def test_history_rejects_gap(self):
        h = valid_history(self.observed, 3)
        self.assertFalse(h.iloc[6])
        self.assertTrue(h.iloc[8])

    def test_outcome_rejects_future_gap(self):
        v = valid_outcome(self.observed, 3)
        self.assertFalse(v.iloc[3])
        self.assertTrue(v.iloc[6])
        self.assertFalse(v.iloc[-1])

    def test_labels_mask_only_unsafe(self):
        y = pd.DataFrame({"y_long_3": [1.] * 12, "y_short_3": [1.] * 12}, index=self.full)
        x = mask_labels(y, self.observed, (3,))
        self.assertTrue(pd.isna(x.iloc[3]["y_long_3"]))
        self.assertEqual(x.iloc[6]["y_long_3"], 1.)

if __name__ == "__main__":
    unittest.main()
