import unittest
import numpy as np
import pandas as pd

from research.intrabar_stress import audit_intrabar_accepted_positions
from research.risk_portfolio import RiskFill, risk_portfolio_audit


class IntrabarStressTests(unittest.TestCase):
    def setUp(self):
        self.index=pd.date_range("2026-07-21T00:00:00Z",periods=20,freq="min")
        self.bars=pd.DataFrame({
            "open":np.full(20,100.),
            "high":np.full(20,100.1),
            "low":np.full(20,99.9),
            "close":np.full(20,100.),
            "source_observed":True,
        },index=self.index)
        self.pos={
            "symbol":"BTCUSDT","side":"long",
            "entry_at":self.index[0].isoformat(),
            "exit_at":self.index[5].isoformat(),
            "entry_notional":5000.,"entry_equity":10000.,
            "effective_leverage":20.,"stop_fraction":.01,
        }

    def audit(self,rows=None,bars=None):
        return audit_intrabar_accepted_positions(
            [self.pos] if rows is None else rows,
            {"BTCUSDT":self.bars if bars is None else bars},
        )

    def test_no_adverse_move_above_threshold_has_zero_liquidation_flags(self):
        result=self.audit()
        self.assertEqual(result["accepted_position_count"],1)
        self.assertEqual(result["positions_with_possible_liquidation"],0)
        self.assertFalse(result["actual_exchange_liquidation_modeled"])
        self.assertFalse(result["portfolio_intraday_mtm_drawdown_computed"])
        self.assertFalse(result["release_approved"])

    def test_ohlc_low_exceeding_isolated_proxy_is_reported_not_claimed_liquidation(self):
        self.bars.loc[self.index[2],"low"]=90.
        report=self.audit()
        self.assertEqual(report["positions_with_possible_liquidation"],1)
        self.assertAlmostEqual(report["worst_adverse_stop_multiple"],10.)
        self.assertAlmostEqual(report["worst_single_position_equity_excursion"],.05)
        self.assertFalse(report["actual_exchange_liquidation_modeled"])

    def test_hypothetical_leverage_grid_keeps_accepted_risk_book_unchanged(self):
        self.bars.loc[self.index[2],"low"]=90.
        result=self.audit()
        scenarios=result["fixed_notional_leverage_scenario_summary"]
        self.assertEqual([x["hypothetical_leverage"] for x in scenarios],
                         [1.,2.,3.,5.,10.,15.,20.])
        self.assertEqual(scenarios[0]["positions_with_possible_proxy_liquidation"],0)
        self.assertEqual(scenarios[-1]["positions_with_possible_proxy_liquidation"],1)
        self.assertEqual(result["accepted_position_count"],1)
        self.assertEqual(result["positions_with_possible_liquidation"],1)
        self.assertFalse(result["release_approved"])

    def test_no_positions_have_zero_hypothetical_liquidations(self):
        result=self.audit(rows=[])
        self.assertTrue(all(
            s["positions_with_possible_proxy_liquidation"] == 0
            for s in result["fixed_notional_leverage_scenario_summary"]
        ))

    def test_short_uses_adverse_high(self):
        self.pos["side"]="short"
        self.bars.loc[self.index[2],"high"]=110.
        report=self.audit()
        self.assertEqual(report["positions_with_possible_liquidation"],1)

    def test_microsecond_resolution_parquet_index_preserves_observed_minute(self):
        micro=self.bars.copy()
        micro.index=micro.index.as_unit("us")
        result=self.audit(bars=micro)
        self.assertEqual(result["accepted_position_count"],1)
        self.assertEqual(result["positions_with_possible_liquidation"],0)

    def test_missing_or_synthetic_minute_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"missing observed minute"):
            self.audit(bars=self.bars.drop(index=self.index[3]))
        bad=self.bars.copy()
        bad.loc[self.index[4],"source_observed"]=False
        with self.assertRaisesRegex(ValueError,"synthetic"):
            self.audit(bars=bad)

    def test_holdout_window_cannot_be_audited(self):
        future=dict(self.pos)
        future["exit_at"]="2026-08-17T00:00:00Z"
        with self.assertRaises(ValueError):
            self.audit(rows=[future])

    def test_empty_trace_remains_unapproved(self):
        x=self.audit(rows=[])
        self.assertEqual(x["accepted_position_count"],0)
        self.assertEqual(x["worst_single_position_equity_excursion"],0)
        self.assertFalse(x["release_approved"])

    def test_trace_contains_only_positions_executed_under_margin_rules(self):
        fill1=RiskFill("BTCUSDT",self.index[0].isoformat(),self.index[5].isoformat(),.01,.01,"long")
        fill2=RiskFill("BTCUSDT",self.index[1].isoformat(),self.index[6].isoformat(),.01,.01,"short")
        result=risk_portfolio_audit([fill1,fill2],record_executed_positions=True)
        self.assertEqual(result["executed_fills"],1)
        self.assertEqual(len(result["accepted_position_trace"]),1)
        self.assertEqual(result["accepted_position_trace"][0]["symbol"],"BTCUSDT")
        self.assertIsNone(risk_portfolio_audit([fill1])["accepted_position_trace"])
        self.assertFalse(result["release_approved"])


if __name__=="__main__":
    unittest.main()
