import unittest

import numpy as np
import pandas as pd

from research.funding_cost import (
    FundingAdjustment, historical_funding_adjustment,
    reconcile_risk_fill_funding,
)
from research.risk_portfolio import RiskFill, risk_portfolio_audit


class FundingAndRiskTests(unittest.TestCase):
    def setUp(self):
        self.t = pd.date_range("2026-07-20T00:00:00Z",periods=36*60+1,freq="min")
        self.bars = pd.DataFrame({"open":np.full(len(self.t),100.),
                                   "source_observed":True},index=self.t)
        ts=pd.to_datetime([
            "2026-07-20T00:00:00Z","2026-07-20T08:00:00Z",
            "2026-07-20T16:00:00Z","2026-07-21T00:00:00Z",
            "2026-07-21T08:00:00Z",
        ])
        self.events=pd.DataFrame(
            {"funding_rate":[0.,.001,0.,0.,0.]}, index=ts)
        self.enter="2026-07-20T06:00:00Z"
        self.exit="2026-07-20T10:00:00Z"

    def cash(self,side):
        return historical_funding_adjustment(
            self.events,self.bars,side=side,
            entry_at=self.enter,exit_at=self.exit)

    def test_funding_positive_rate_long_pays_short_receives(self):
        a=self.cash("long")
        b=self.cash("short")
        self.assertEqual(a.settled_funding_events,1)
        self.assertAlmostEqual(a.net_funding_fraction_of_entry_notional,-.001)
        self.assertAlmostEqual(b.net_funding_fraction_of_entry_notional,.001)
        self.assertAlmostEqual(a.payments_fraction,.001)
        self.assertAlmostEqual(b.receipts_fraction,.001)

    def test_negative_funding_swaps_cashflow_direction(self):
        self.events.loc[pd.Timestamp("2026-07-20T08:00:00Z"),"funding_rate"]=-.002
        self.assertAlmostEqual(self.cash("long").net_funding_fraction_of_entry_notional,.002)
        self.assertAlmostEqual(self.cash("short").net_funding_fraction_of_entry_notional,-.002)

    def test_settlement_at_entry_excluded_exit_included(self):
        f=historical_funding_adjustment(
            self.events,self.bars,side="long",
            entry_at="2026-07-20T08:00:00Z",exit_at="2026-07-20T08:00:00Z")
        self.assertEqual(f.settled_funding_events,0)
        g=historical_funding_adjustment(
            self.events,self.bars,side="long",
            entry_at=self.enter,exit_at="2026-07-20T08:00:00Z")
        self.assertEqual(g.settled_funding_events,1)

    def test_funding_is_scaled_by_observed_price_proxy(self):
        self.bars.loc[pd.Timestamp("2026-07-20T08:00:00Z"),"open"]=110.
        self.assertAlmostEqual(self.cash("long").net_funding_fraction_of_entry_notional,-.0011)

    def test_millisecond_settlement_uses_same_observed_minute_only(self):
        exact=pd.Timestamp("2026-07-20T08:00:00Z")
        fractional=exact+pd.Timedelta(milliseconds=4)
        self.events=self.events.rename(index={exact:fractional}).sort_index()
        self.assertAlmostEqual(self.cash("long").net_funding_fraction_of_entry_notional,-.001)
        missing=self.bars.drop(index=exact)
        with self.assertRaisesRegex(ValueError, "funding settlement minute"):
            historical_funding_adjustment(
                self.events,missing,side="long",entry_at=self.enter,exit_at=self.exit)

    def test_unobserved_or_missing_settlement_bar_fails_closed(self):
        t=pd.Timestamp("2026-07-20T08:00:00Z")
        self.bars.loc[t,"source_observed"]=False
        with self.assertRaises(ValueError):
            self.cash("long")
        self.bars.loc[t,"source_observed"]=True
        with self.assertRaises(ValueError):
            historical_funding_adjustment(self.events,self.bars.drop(index=t),
                side="short",entry_at=self.enter,exit_at=self.exit)

    def test_archival_gap_or_incomplete_coverage_fails_closed(self):
        with self.assertRaises(ValueError):
            historical_funding_adjustment(
                self.events.iloc[1:],self.bars,side="long",
                entry_at=self.enter,exit_at=self.exit)
        missing=self.events.drop(index=self.events.index[1:4])
        with self.assertRaises(ValueError):
            historical_funding_adjustment(
                missing,self.bars,side="long",entry_at=self.enter,exit_at=self.exit)

    def test_funding_explicitly_required_and_not_double_charged(self):
        fill=RiskFill("BTCUSDT",self.enter,self.exit,gross_return=.01,
                     stop_fraction=.01,side="long")
        with self.assertRaisesRegex(ValueError,"funding"):
            risk_portfolio_audit([fill],require_funding_reconciled=True)
        reconciled=reconcile_risk_fill_funding(fill,self.events,self.bars)
        with self.assertRaisesRegex(ValueError,"already reconciled"):
            reconcile_risk_fill_funding(reconciled,self.events,self.bars)
        funded=risk_portfolio_audit([reconciled],require_funding_reconciled=True)
        old=risk_portfolio_audit([fill])
        self.assertEqual(funded["executed_fills"],1)
        self.assertLess(funded["net_equity_return"],old["net_equity_return"])
        self.assertLess(funded["realized_funding_pnl_cash"],0)
        self.assertTrue(funded["funding_reconciled_for_all_fills"])
        self.assertFalse(funded["exchange_liquidation_engine_simulated"])
        self.assertFalse(funded["release_approved"])

    def test_short_funding_receipt_increases_cash(self):
        fill=RiskFill("BTCUSDT",self.enter,self.exit,.01,.01,"short")
        reconciled=reconcile_risk_fill_funding(fill,self.events,self.bars)
        funded=risk_portfolio_audit([reconciled],require_funding_reconciled=True)
        old=risk_portfolio_audit([fill])
        self.assertGreater(funded["net_equity_return"],old["net_equity_return"])

    def test_invalid_rate_and_holdout_are_rejected(self):
        self.events.loc[self.events.index[1],"funding_rate"]=np.nan
        with self.assertRaises(ValueError):
            self.cash("long")
        with self.assertRaises(ValueError):
            historical_funding_adjustment(self.events,self.bars,side="long",
                entry_at="2026-08-17T00:00:00Z",
                exit_at="2026-08-17T01:00:00Z")


if __name__=="__main__":
    unittest.main()
