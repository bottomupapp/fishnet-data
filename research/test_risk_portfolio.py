import unittest

import pandas as pd

from research.execution import net_after_costs, net_after_costs_short
from research.risk_portfolio import RiskFill, risk_portfolio_audit


BASE = pd.Timestamp("2026-07-20T00:00:00Z")


def trade(symbol="BTCUSDT", at=0, gross=.001, stop=.01, duration=5):
    enter = BASE + pd.Timedelta(minutes=at)
    leave = enter + pd.Timedelta(minutes=duration)
    return RiskFill(symbol, enter.isoformat(), leave.isoformat(), gross, stop)


class RiskPortfolioTests(unittest.TestCase):
    def test_full_planned_stop_including_costs_is_one_percent_of_equity(self):
        out = risk_portfolio_audit([trade(gross=-.01, stop=.01)])
        self.assertAlmostEqual(out["net_equity_return"], -.01, places=10)
        self.assertAlmostEqual(out["mean_realized_r_on_planned_stop_risk"], -1, places=10)

    def test_net_three_point_two_r_at_one_percent_risk_grows_equity_3_2_percent(self):
        stop = .01
        retention = 1 + net_after_costs(0., 6., 3.)
        stop_loss = -net_after_costs(-stop, 6., 3.)
        gross_target = (1 + 3.2 * stop_loss) / retention - 1
        out = risk_portfolio_audit([trade(gross=gross_target, stop=stop)])
        self.assertAlmostEqual(out["net_equity_return"], .032, places=10)
        self.assertAlmostEqual(out["mean_realized_r_on_planned_stop_risk"], 3.2, places=10)
        self.assertFalse(out["release_approved"])

    def test_short_planned_stop_including_costs_is_one_percent(self):
        position = trade(gross=-.01, stop=.01)
        position = RiskFill(position.symbol, position.entry_at, position.exit_at,
                            position.gross_return, position.stop_fraction, side="short")
        out = risk_portfolio_audit([position])
        self.assertAlmostEqual(out["net_equity_return"], -.01, places=10)
        self.assertAlmostEqual(out["mean_realized_r_on_planned_stop_risk"], -1, places=10)

    def test_short_risk_budget_three_r_net(self):
        fee, slip = 6., 3.
        stop=.008
        cost_loss = -net_after_costs_short(-stop, fee, slip)
        wanted = 3.0 * cost_loss
        sell=(1 - fee/10000)*(1 - slip/10000)
        buy=(1 + fee/10000)*(1 + slip/10000)
        gross = 1 - (1 - wanted) * sell / buy
        base = trade(gross=gross, stop=stop)
        short = RiskFill(base.symbol,base.entry_at,base.exit_at,
                         base.gross_return,base.stop_fraction,side="short")
        out=risk_portfolio_audit([short])
        self.assertAlmostEqual(out["net_equity_return"],.03,places=9)

    def test_unrecognized_trade_side_fails_closed(self):
        base=trade()
        with self.assertRaises(ValueError):
            risk_portfolio_audit([RiskFill(base.symbol,base.entry_at,base.exit_at,
                                           base.gross_return,base.stop_fraction,side="unknown")])

    def test_more_leverage_does_not_increase_return_without_exposure_change(self):
        item = [trade(gross=.02, stop=.01)]
        a = risk_portfolio_audit(item, max_leverage=10)
        b = risk_portfolio_audit(item, max_leverage=50)
        self.assertAlmostEqual(a["net_equity_return"], b["net_equity_return"], places=12)

    def test_gross_exposure_cap_reduces_instead_of_increasing_risk(self):
        out = risk_portfolio_audit(
            [trade(stop=.0005, gross=.01)],
            max_gross_exposure=1, max_leverage=20,
        )
        self.assertLessEqual(out["max_active_gross_exposure_to_equity"], 1.00000001)
        self.assertLess(out["mean_realized_r_on_planned_stop_risk"], 20)

    def test_concurrent_long_and_short_on_distinct_majors(self):
        a = trade("BTCUSDT", at=0, gross=.01, stop=.01)
        b = trade("ETHUSDT", at=0, gross=.01, stop=.01)
        opposite = RiskFill(b.symbol, b.entry_at, b.exit_at,
                            b.gross_return, b.stop_fraction, side="short")
        result = risk_portfolio_audit([a, opposite], max_concurrent_positions=2)
        self.assertEqual(result["executed_fills"], 2)
        self.assertEqual(result["skipped_fills"]["capacity"], 0)
        self.assertLessEqual(result["max_simultaneous_planned_risk_fraction_of_equity"], .0200001)
        self.assertFalse(result["release_approved"])

    def test_opposite_direction_same_symbol_requires_explicit_hedge_mode(self):
        long_fill = trade("BTCUSDT", at=0, gross=.01, stop=.01)
        short_fill = RiskFill(long_fill.symbol, long_fill.entry_at, long_fill.exit_at,
                              .01, .01, side="short")
        result = risk_portfolio_audit([long_fill, short_fill], max_concurrent_positions=2)
        self.assertEqual(result["executed_fills"], 1)
        self.assertEqual(result["skipped_fills"]["symbol_overlap"], 1)

    def test_same_minute_same_symbol_predeclared_priority_never_uses_future_exit(self):
        # Predeclared strategy priority is input order: LONG first. At decision
        # time neither future holding period nor gross return is observable.
        entry=BASE.isoformat()
        for long_minutes,short_minutes in ((8,1),(1,8)):
            long_fill=RiskFill(
                "BTCUSDT",entry,
                (BASE+pd.Timedelta(minutes=long_minutes)).isoformat(),
                gross_return=-.002, stop_fraction=.01, side="long",
            )
            short_fill=RiskFill(
                "BTCUSDT",entry,
                (BASE+pd.Timedelta(minutes=short_minutes)).isoformat(),
                gross_return=.025,stop_fraction=.01,side="short",
            )
            book=risk_portfolio_audit([long_fill,short_fill],
                                       record_executed_positions=True)
            only_first=risk_portfolio_audit([long_fill])
            self.assertEqual(book["executed_fills"],1)
            self.assertEqual(book["skipped_fills"]["symbol_overlap"],1)
            self.assertEqual(book["accepted_position_trace"][0]["side"],"long")
            self.assertAlmostEqual(book["net_equity_return"],
                                   only_first["net_equity_return"],places=12)
            self.assertFalse(book["release_approved"])

    def test_overlap_and_capacity_skip_additional_positions(self):
        fills = [trade(at=0), trade(at=1), trade("ETHUSDT", at=1)]
        a = risk_portfolio_audit(fills, max_concurrent_positions=1)
        self.assertEqual(a["executed_fills"], 1)
        self.assertEqual(a["skipped_fills"]["symbol_overlap"], 1)
        self.assertEqual(a["skipped_fills"]["capacity"], 1)

    def test_exit_minute_cannot_reenter_until_following_minute(self):
        a = risk_portfolio_audit([
            trade(at=0, duration=2),
            trade(at=2, duration=2),
            trade(at=3, duration=2),
        ])
        self.assertEqual(a["executed_fills"], 2)
        self.assertEqual(a["skipped_fills"]["symbol_overlap"], 1)

    def test_gap_through_stop_can_lose_more_than_one_percent(self):
        a = risk_portfolio_audit([trade(gross=-.08, stop=.01)])
        self.assertLess(a["net_equity_return"], -.01)
        self.assertLess(a["mean_realized_r_on_planned_stop_risk"], -1)

    def test_too_tight_margin_proxy_restricts_effective_leverage(self):
        a = risk_portfolio_audit([trade(stop=.02)], max_leverage=100)
        self.assertLess(a["highest_effective_leverage_used"], 40)

    def test_negative_equity_and_naive_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            risk_portfolio_audit([], starting_capital=0)
        with self.assertRaises(ValueError):
            risk_portfolio_audit([RiskFill("BTC", "2026-01-01", "2026-01-02", .1, .01)])
        with self.assertRaises(ValueError):
            risk_portfolio_audit([trade(gross=float("nan"))])

    def test_empty_strategy_cannot_pass_net_seven_percent_target(self):
        a = risk_portfolio_audit([])
        self.assertEqual(a["net_equity_return"], 0)
        self.assertFalse(a["net_30d_goal_met_in_development"])
        self.assertFalse(a["release_approved"])


if __name__ == "__main__":
    unittest.main()
