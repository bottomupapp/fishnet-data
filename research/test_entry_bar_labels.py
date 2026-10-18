import unittest
import numpy as np
import pandas as pd
from research.features import labels

class EntryBarLabels(unittest.TestCase):
    def test_entry_minute_barrier_counts(self):
        idx=pd.date_range("2026-01-01",periods=70,freq="min",tz="UTC")
        bars=pd.DataFrame({"open":np.full(70,100.),"high":np.full(70,100.),"low":np.full(70,100.)},index=idx)
        bars.iloc[1,bars.columns.get_loc("high")]=101.
        sigma=pd.Series(np.full(70,0.00001),index=idx)
        out=labels(bars,sigma)
        self.assertEqual(out.iloc[0]["y_long_5"],1.)
        self.assertEqual(out.iloc[0]["y_short_5"],0.)
    def test_simultaneous_barrier_touch_is_ambiguous(self):
        idx=pd.date_range("2026-01-01",periods=70,freq="min",tz="UTC")
        bars=pd.DataFrame({"open":np.full(70,100.),"high":np.full(70,100.),"low":np.full(70,100.)},index=idx)
        bars.iloc[1,bars.columns.get_loc("high")]=101.
        bars.iloc[1,bars.columns.get_loc("low")]=99.
        out=labels(bars,pd.Series(np.full(70,0.00001),index=idx))
        self.assertEqual(out.iloc[0]["y_long_5"],0.)
        self.assertEqual(out.iloc[0]["y_short_5"],0.)

if __name__=="__main__": unittest.main()
