import unittest

from research.okx_public_tiers import parse_public_swap_tiers

URL = ("https://www.okx.com/api/v5/public/position-tiers?"
       "instType=SWAP&tdMode=isolated&instFamily=BTC-USDT")


def fixture():
    # Synthetic API-shaped tiers, NOT actual OKX leverage or margin rates.
    return {"code": "0", "data": [
        {"instFamily": "BTC-USDT", "tier": "1", "minSz": "0",
         "maxSz": "100", "mmr": "0.005", "imr": "0.05", "maxLever": "20"},
        {"instFamily": "BTC-USDT", "tier": "2", "minSz": "100",
         "maxSz": "200", "mmr": "0.01", "imr": "0.1", "maxLever": "10"},
    ]}


class PublicOkxTierTests(unittest.TestCase):
    def parse(self, response=None, url=URL):
        return parse_public_swap_tiers(
            fixture() if response is None else response,
            inst_family="BTC-USDT", captured_at_utc="2026-10-22T00:00:00Z",
            source_url=url,
        )

    def test_keeps_contracts_separate_from_usdt_notional(self):
        result = self.parse()
        self.assertEqual(result["tiers"][0]["max_contracts"], "100")
        self.assertEqual(result["tier_size_unit"], "contracts_NOT_USDT")
        self.assertFalse(result["usdt_notional_comparison_available"])
        self.assertFalse(result["verified_at_historical_fishnet_event_time"])
        self.assertFalse(result["release_approved"])

    def test_failure_responses_or_foreign_instruments_rejected(self):
        for kind in ("code", "instFamily", "tier", "maxSz", "mmr", "imr", "maxLever"):
            response = fixture()
            if kind == "code":
                response["code"] = "51000"
            else:
                response["data"][0][kind] = ("ETH-USDT" if kind == "instFamily" else "not-a-number")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.parse(response)

    def test_rejects_nonmonotone_contract_bounds_and_rates(self):
        for key, value in (
            ("minSz", "0"), ("maxSz", "50"), ("mmr", "0.001"),
            ("imr", "0.005"), ("maxLever", "30"),
        ):
            response = fixture()
            response["data"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(response)

    def test_rejects_wrong_url_mode_and_host(self):
        for url in (
            URL.replace("isolated", "cross"),
            URL.replace("www.okx.com", "www.okx.com.evil.net"),
            URL.replace("BTC-USDT", "ETH-USDT"),
            URL.replace("https:", "http:"),
            "https://www.okx.com/api/v5/account/position-tiers?"
            "instType=SWAP&tdMode=isolated&instFamily=BTC-USDT",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.parse(url=url)

    def test_missing_and_naive_capture_time_refused(self):
        with self.assertRaises(ValueError):
            parse_public_swap_tiers(
                fixture(), inst_family="BTC-USDT",
                captured_at_utc="2026-10-22T00:00:00", source_url=URL)


if __name__ == "__main__":
    unittest.main()
