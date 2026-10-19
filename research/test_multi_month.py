import unittest

from research.multi_month import FOLDS, SCENARIOS, summarize


def report(start: str, mean: float = .0001, selected: int = 25) -> dict:
    return {
        "test_start": f"{start} 00:00:00+00:00",
        "test_end_exclusive": "2026-08-16 00:00:00+00:00",
        "rows": {"test": 8640},
        "research_abstention": {
            "release_approved": False,
            "gate_admitted": selected,
            "admitted_simulated": selected,
            "mean_net_return_per_admitted_trade": mean,
            "gate_decisions": {"research_candidate_only": selected},
        },
        "execution_audit": {
            "status": "illustrative_cost_assumptions_not_live_execution",
            "selected": selected,
            "simulated": selected,
            "cost_sensitivity": [
                {"scenario": name, "mean_net_return_per_simulated_trade": mean,
                 "net_positive_fraction": .60} for name in SCENARIOS
            ],
        },
    }


class MultiMonthReleaseSafetyTests(unittest.TestCase):
    def test_never_auto_approves_even_if_all_folds_positive(self):
        summary = summarize([report(x) for x in FOLDS])
        self.assertTrue(summary["development_consistency_gate_passed"])
        self.assertFalse(summary["release_approved"])

    def test_one_negative_month_fails_development_gate(self):
        reports = [report(x) for x in FOLDS]
        reports[2] = report(FOLDS[2], mean=-.001)
        self.assertFalse(summarize(reports)["development_consistency_gate_passed"])

    def test_small_signal_sample_fails_development_gate(self):
        reports = [report(x) for x in FOLDS]
        reports[0] = report(FOLDS[0], selected=12)
        self.assertFalse(summarize(reports)["development_consistency_gate_passed"])

    def test_gate_cannot_approve_negative_month(self):
        reports = [report(x) for x in FOLDS]
        reports[1]["research_abstention"]["mean_net_return_per_admitted_trade"] = -.0005
        result = summarize(reports)
        self.assertFalse(result["abstention_development_consistency_gate_passed"])
        self.assertFalse(result["release_approved"])

    def test_gate_refuses_invalid_count(self):
        reports = [report(x) for x in FOLDS]
        reports[1]["research_abstention"]["gate_admitted"] = 26
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_missing_month_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([report(x) for x in FOLDS[:3]])

    def test_missing_cost_scenario_is_rejected(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["execution_audit"]["cost_sensitivity"].pop()
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_mismatched_simulation_count_is_rejected(self):
        reports = [report(x) for x in FOLDS]
        reports[0]["execution_audit"]["simulated"] -= 1
        with self.assertRaises(ValueError):
            summarize(reports)


if __name__ == "__main__":
    unittest.main()
