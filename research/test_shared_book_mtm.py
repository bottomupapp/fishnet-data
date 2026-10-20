import unittest
import numpy as np
import pandas as pd

from research.shared_book_mtm import audit_shared_book_minute_mtm
from research.risk_portfolio import RiskFill, risk_portfolio_audit


class SharedBookMtmTests(unittest.TestCase):
    def setUp(self):
        self.t = pd.Timestamp("2026-07-21T00:00:00Z")
        self.index = pd.date_range(self.t, periods=8, freq="min")
        self.bars = pd.DataFrame({
            "open": np.full(8, 100.),
            "high": np.full(8, 100.),
            "low": np.full(8, 100.),
            "close": np.full(8, 100.),
            "source_observed": True,
        }, index=self.index)

    def trace(self, side="long", gross=0.):
        fill=RiskFill(
            "BTCUSDT", self.t.isoformat(),
            (self.t + pd.Timedelta(minutes=4)).isoformat(),
            gross, .01, side,
        )
        book=risk_portfolio_audit([fill], record_executed_positions=True)
        return book["accepted_position_trace"], book

    def audit(self, trace, bars=None):
        return audit_shared_book_minute_mtm(
            trace, {"BTCUSDT": self.bars if bars is None else bars})

    def test_flat_price_reconciles_including_two_sided_costs(self):
        trace, book = self.trace()
        result=self.audit(trace)
        self.assertAlmostEqual(result["last_settled_equity"],book["final_equity"])
        self.assertEqual(result["accepted_position_count"],1)
        self.assertTrue(result["full_close_settlement_cash_reconciled"])
        self.assertFalse(result["release_approved"])

    def test_unrealized_adverse_long_loss_is_visible_before_profitable_exit(self):
        trace, book=self.trace(gross=.02)
        bars=self.bars.copy()
        bars.loc[self.index[2],"low"]=90.
        bars.loc[self.index[2],"close"]=95.
        result=self.audit(trace,bars)
        self.assertGreater(result["worst_minute_close_drawdown"],0.)
        self.assertGreater(result["worst_simultaneous_intraminute_proxy_drawdown"],
                           result["worst_minute_close_drawdown"])
        self.assertAlmostEqual(result["last_settled_equity"],book["final_equity"])

    def test_short_high_is_adverse_even_if_final_settlement_profitable(self):
        trace, _=self.trace(side="short", gross=.01)
        bars=self.bars.copy()
        bars.loc[self.index[1],"high"]=110.
        bars.loc[self.index[1],"close"]=105.
        result=self.audit(trace,bars)
        self.assertGreater(result["worst_simultaneous_intraminute_proxy_drawdown"],0.)

    def test_fails_closed_on_missing_or_synthetic_minutes(self):
        trace, _=self.trace()
        with self.assertRaisesRegex(ValueError,"missing observed"):
            self.audit(trace,self.bars.drop(self.index[2]))
        bad=self.bars.copy()
        bad.loc[self.index[3],"source_observed"]=False
        with self.assertRaisesRegex(ValueError,"synthetic"):
            self.audit(trace,bad)

    def test_holdout_and_empty_book(self):
        self.assertFalse(self.audit([])["release_approved"])
        trace, _=self.trace()
        trace[0]["exit_at"]="2026-08-17T00:00:00Z"
        with self.assertRaises(ValueError):
            self.audit(trace)

    def test_microsecond_archive_timestamps_are_accepted(self):
        trace,_=self.trace()
        bars=self.bars.copy()
        bars.index=bars.index.as_unit("us")
        self.assertEqual(self.audit(trace,bars)["accepted_position_count"],1)


if __name__ == "__main__":
    unittest.main()
