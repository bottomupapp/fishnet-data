"""Research-only, ex-ante cost-aware abstention for Fishnet LONG hypotheses.

This pure function does not inspect future bars or outcome labels. It is a
predeclared heuristic, NOT a calibrated expected-return model: LightGBM's
probability of a t+1 barrier hit may differ from the chance of a profitable
t+2 delayed execution. It never approves a live signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    estimated_gross_surplus: float | None
    cost_fraction: float | None
    margin_after_cost: float | None


def evaluate_admission(
    probability: float,
    barrier_fraction: float,
    *,
    fee_bps_per_side: float = 6.0,
    slippage_bps_per_side: float = 3.0,
    uncertainty_buffer_bps: float = 5.0,
) -> Admission:
    """Gate LONG-15 candidates using only prediction-time features and costs.

    Conservative label proxy: a successful first-barrier outcome gains the
    historical barrier, and EVERY other outcome loses an equal barrier amount.
    This proxy is not a real expected PnL because delayed fills, timeouts,
    funding and gap stops can differ from the barrier label.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return Admission(False, "invalid_probability", None, None, None)
    if not math.isfinite(barrier_fraction) or not 0 < barrier_fraction < 1:
        return Admission(False, "invalid_barrier", None, None, None)
    costs = (fee_bps_per_side, slippage_bps_per_side, uncertainty_buffer_bps)
    if not all(math.isfinite(x) and 0 <= x < 10000 for x in costs):
        raise ValueError("invalid assumed costs or uncertainty buffer")
    fee = fee_bps_per_side / 10000
    slip = slippage_bps_per_side / 10000
    # Flat-price round-trip fractional cost (fees and slippage, both sides).
    retention = ((1 - fee) * (1 - slip)) / ((1 + fee) * (1 + slip))
    cost_fraction = 1 - retention
    gross_proxy = (2 * probability - 1) * barrier_fraction
    # Require positive proxy surplus after all costs and a buffer for model error.
    margin = (1 + gross_proxy) * retention - 1 - uncertainty_buffer_bps / 10000
    if margin <= 0:
        return Admission(False, "estimated_edge_below_cost", gross_proxy, cost_fraction, margin)
    return Admission(True, "research_candidate_only", gross_proxy, cost_fraction, margin)


def minimum_barrier_to_beat_cost(
    *,
    fee_bps_per_side: float = 6.0,
    slippage_bps_per_side: float = 3.0,
    uncertainty_buffer_bps: float = 5.0,
) -> float:
    """Smallest favorable price move feasible at perfect label accuracy.

    This is a threshold of the symmetric-barrier *proxy*, not a market
    profitability forecast; real t+2 fills, stop gaps and funding differ.
    """
    costs = (fee_bps_per_side, slippage_bps_per_side, uncertainty_buffer_bps)
    if not all(math.isfinite(x) and 0 <= x < 10000 for x in costs):
        raise ValueError("invalid costs")
    fee, slip = fee_bps_per_side / 10000, slippage_bps_per_side / 10000
    retention = (1 - fee) * (1 - slip) / ((1 + fee) * (1 + slip))
    return float((1 + uncertainty_buffer_bps / 10000) / retention - 1)


def minimum_probability_for_cost_surplus(
    barrier_fraction: float,
    *,
    fee_bps_per_side: float = 6.0,
    slippage_bps_per_side: float = 3.0,
    uncertainty_buffer_bps: float = 5.0,
) -> float:
    """Exact break-even hit probability under the gate's symmetric barrier proxy.

    This explains whether *any* probability in [0,1] could pass. It is NOT
    an observed hit rate or an estimated return. No labels or future bars.
    """
    if not math.isfinite(barrier_fraction) or not 0 < barrier_fraction < 1:
        raise ValueError("invalid barrier")
    required_gross = minimum_barrier_to_beat_cost(
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        uncertainty_buffer_bps=uncertainty_buffer_bps,
    )
    return float((1 + required_gross / barrier_fraction) / 2)
