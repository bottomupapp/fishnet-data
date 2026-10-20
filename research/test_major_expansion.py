import unittest
import pandas as pd
from research.major_expansion import MAJORS, LABELS, audit_features


class MajorExpansionTests(unittest.TestCase):
    def make_df(self):
        dates=pd.date_range("2026-03-01",periods=30001,freq="5min",tz="UTC")
        parts=[]
        for sym in MAJORS:
            data={"symbol":[sym]*len(dates),"t":dates}
            data.update({name:[0.]*len(dates) for name in LABELS})
            parts.append(pd.DataFrame(data))
        return pd.concat(parts,ignore_index=True)

    def test_five_major_label_coverage_passes_but_not_live_approved(self):
        result=audit_features(self.make_df())
        self.assertEqual(len(result["symbols"]),5)
        self.assertFalse(result["model_live_approved"])
        self.assertFalse(result["training_universe_approved"])

    def test_duplicate_minute_rejected(self):
        data=self.make_df()
        data=pd.concat([data,data.iloc[:1]],ignore_index=True)
        with self.assertRaises(ValueError):
            audit_features(data)

    def test_missing_symbol_rejected(self):
        with self.assertRaises(ValueError):
            audit_features(self.make_df().query("symbol != 'XRPUSDT'"))

    def test_missing_label_history_rejected(self):
        data=self.make_df()
        data.loc[data.symbol=="SOLUSDT","y_short_60"]=float("nan")
        with self.assertRaises(ValueError):
            audit_features(data)

    def test_holdout_only_rejected(self):
        data=self.make_df()
        data["t"]=pd.to_datetime(data.t,utc=True)+pd.Timedelta(days=200)
        with self.assertRaises(ValueError):
            audit_features(data)

if __name__=="__main__":unittest.main()
