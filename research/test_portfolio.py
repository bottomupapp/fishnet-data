import unittest

from research.portfolio import CandidateFill, NET_30D_TARGET, portfolio_audit


def trade(symbol, minute, rate, duration=5):
    from pandas import Timestamp, Timedelta
    start = Timestamp("2026-07-20T00:00:00Z") + Timedelta(minutes=minute)
    return CandidateFill(symbol, start.isoformat(), (start + Timedelta(minutes=duration)).isoformat(), rate)


class PortfolioAuditTests(unittest.TestCase):
    def test_empty_window_is_zero_not_seven_percent(self):
        report = portfolio_audit([])
        self.assertEqual(report["net_equity_return"], 0)
        self.assertFalse(report["net_30d_goal_met_in_development"])
        self.assertFalse(report["release_approved"])

    def test_net_return_is_on_staked_not_total_capital(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02)])
        self.assertAlmostEqual(report["net_equity_return"], .002, places=12)
        self.assertEqual(report["executed_fills"], 1)

    def test_compounding_when_positions_settle(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02), trade("BTCUSDT", 8, .02)])
        self.assertAlmostEqual(report["net_equity_return"], 1.002 ** 2 - 1, places=12)

    def test_overlapping_same_symbol_is_rejected(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02), trade("BTCUSDT", 3, .02)])
        self.assertEqual(report["executed_fills"], 1)
        self.assertEqual(report["skipped_fills"]["symbol_overlap"], 1)

    def test_capacity_rejects_second_symbol_without_leverage(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02), trade("ETHUSDT", 0, .02)], max_concurrent_positions=1)
        self.assertEqual(report["executed_fills"], 1)
        self.assertEqual(report["skipped_fills"]["capacity"], 1)

    def test_exit_minute_is_still_reserved(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02, duration=2), trade("BTCUSDT", 2, .02)])
        self.assertEqual(report["executed_fills"], 1)

    def test_same_minute_exit_is_valid_but_capacity_remains_reserved(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .02, duration=0), trade("ETHUSDT", 0, .02, duration=0)], max_concurrent_positions=1)
        self.assertEqual(report["executed_fills"], 1)

    def test_losses_reduce_equity_and_increase_realized_drawdown(self):
        report = portfolio_audit([trade("BTCUSDT", 0, -.05)], stake_fraction=.5)
        self.assertAlmostEqual(report["net_equity_return"], -.025)
        self.assertGreater(report["max_realized_drawdown"], 0)

    def test_seven_percent_is_goal_not_release(self):
        report = portfolio_audit([trade("BTCUSDT", 0, .08)], stake_fraction=1.0)
        self.assertEqual(NET_30D_TARGET, .07)
        self.assertTrue(report["net_30d_goal_met_in_development"])
        self.assertFalse(report["release_approved"])

    def test_invalid_timestamps_and_returns_fail_closed(self):
        bad = [CandidateFill("BTCUSDT", "2026-01-01", "2026-01-02", .01),
               trade("BTCUSDT", 0, float("nan")),
               trade("BTCUSDT", 0, -1.1)]
        for fill in bad:
            with self.assertRaises(ValueError):
                portfolio_audit([fill])

    def test_invalid_sizing_fails_closed(self):
        with self.assertRaises(ValueError):
            portfolio_audit([], stake_fraction=1.1)
        with self.assertRaises(ValueError):
            portfolio_audit([], max_concurrent_positions=0)


if __name__ == "__main__":
    unittest.main()
