import unittest

from research.okx_margin_reference import (
    compare_current_okx_reference, validate_okx_tier_snapshot,
)


def snapshot():
    # Synthetic fixture, NOT actual OKX tier values.
    return {
        "exchange": "OKX",
        "instrument": "BTC-USDT-SWAP",
        "margin_mode": "isolated",
        "source": "OKX_official_tier_reference",
        "source_url": "https://www.okx.com/docs-v5/en/",
        "captured_at_utc": "2026-10-21T00:00:00Z",
        "tiers": [
            {"tier": 1, "max_notional_usdt": 10000,
             "maintenance_margin_rate": .01, "max_leverage": 20},
            {"tier": 2, "max_notional_usdt": 50000,
             "maintenance_margin_rate": .02, "max_leverage": 10},
        ],
    }


class OkxMarginReferenceTests(unittest.TestCase):
    def test_normalized_current_reference_never_claims_historical_liquidation(self):
        reference = validate_okx_tier_snapshot(snapshot())
        self.assertFalse(reference["verified_historical_at_fishnet_event_time"])
        self.assertFalse(reference["exchange_liquidation_engine_simulated"])
        r = compare_current_okx_reference(snapshot(), notional_usdt=25000, leverage=15)
        self.assertEqual(r["selected_reference_tier"], 2)
        self.assertTrue(r["hypothetical_leverage_above_current_tier_limit"])
        self.assertIsNone(r["historical_liquidation_result"])
        self.assertFalse(r["release_approved"])

    def test_missing_or_unsourced_tiers_fail_closed(self):
        for field, value in (
            ("tiers", []), ("margin_mode", "cross"),
            ("source", "guess"), ("instrument", "BTCUSDT"),
            ("captured_at_utc", "2026-10-21T00:00:00"),
            ("source_url", "https://example.com/"),
        ):
            record = snapshot()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_okx_tier_snapshot(record)

    def test_nonmonotone_or_incomplete_tiers_rejected(self):
        for key, value in (
            ("max_notional_usdt", 9000), ("maintenance_margin_rate", .005),
            ("max_leverage", 1000), ("tier", 3),
        ):
            record = snapshot()
            record["tiers"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_okx_tier_snapshot(record)

    def test_position_outside_sourced_range_refuses_to_guess(self):
        with self.assertRaisesRegex(ValueError, "outside sourced tier"):
            compare_current_okx_reference(snapshot(), notional_usdt=100000, leverage=5)


if __name__ == "__main__":
    unittest.main()
