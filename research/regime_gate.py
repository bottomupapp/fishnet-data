"""Fishnet ex-ante regime diagnostics and pretest net-R admission study.

Four predeclared regimes: the 240-minute trend either agrees or disagrees
with trade direction, crossed with 60-minute realized volatility above/below
the trailing 1-day reference. A past-volume z-score below -1 (or missing
regime evidence) is excluded; volume is only an activity proxy, NOT measured
order-book liquidity. No test outcomes determine a regime or score threshold.

This is a multiple-hypothesis development experiment. The 3 standard-error
haircut is HEURISTIC, not a rigorous dependent-event confidence bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from research.run import dedupe

REGIMES = ("aligned_calm", "aligned_fast", "counter_calm", "counter_fast")
ACTIVITY_Z_FLOOR = -1.0
TOP_FRACTION = 0.20
MIN_PER_REGIME = 200
MIN_TOP_EVENTS = 30
UNCERTAINTY_MULTIPLE = 3.0
ECONOMIC_BUFFER_R = .10


@dataclass(frozen=True)
class RegimeEvidence:
    regime: str
    model_score_threshold: float | None
    calibration_rows: int
    nonoverlapping_top_events: int
    mean_realized_net_r: float | None
    conservative_net_r_proxy: float | None
    eligible_for_research: bool


def regime_of(frame: pd.DataFrame, *, side: str) -> pd.Series:
    """Assign only from causal, pre-entry return/vol/activity features."""
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    required = {"ret_240", "rv_60", "vol15_z"}
    if not required.issubset(frame.columns):
        raise ValueError("missing causal trend, volatility, or activity features")
    inputs = frame[["ret_240", "rv_60", "vol15_z"]].to_numpy(dtype=float)
    valid = np.isfinite(inputs).all(axis=1)
    valid &= inputs[:, 1] > 0
    valid &= inputs[:, 2] >= ACTIVITY_Z_FLOOR
    signed = inputs[:, 0] * (1 if side == "long" else -1)
    aligned = signed > 0
    fast = inputs[:, 1] >= 1.0
    names = np.where(aligned, "aligned_", "counter_")
    suffixes = np.where(fast, "fast", "calm")
    return pd.Series(np.where(valid, np.char.add(names, suffixes), "excluded"),
                     index=frame.index, name="regime")


def fit_regime_gates(
    calibration: pd.DataFrame, raw_predictions: np.ndarray,
    *, side: str, horizon_minutes: int,
) -> dict[str, RegimeEvidence]:
    """Fit 4 independently predeclared regime gates on PRE-test realized R.

    High-score candidates are selected before within-symbol time de-overlap
    so an early low-score row does not block a later high-score candidate.
    """
    if not {"t", "symbol", "net_r"}.issubset(calibration.columns):
        raise ValueError("dated realized net-R calibration required")
    if type(horizon_minutes) is not int or not 1 <= horizon_minutes <= 240:
        raise ValueError("invalid horizon")
    scores = np.asarray(raw_predictions, dtype=float)
    if (scores.shape != (len(calibration),) or not np.isfinite(scores).all()
        or not np.isfinite(calibration["net_r"].to_numpy(dtype=float)).all()):
        raise ValueError("invalid calibration scores or outcomes")
    cal = calibration.copy()
    cal["score"] = scores
    cal["regime"] = regime_of(cal, side=side)
    evidence = {}
    for name in REGIMES:
        slice_ = cal[cal.regime == name]
        n = len(slice_)
        if n < MIN_PER_REGIME:
            evidence[name] = RegimeEvidence(name, None, n, 0, None, None, False)
            continue
        threshold = float(np.quantile(slice_.score.to_numpy(), 1 - TOP_FRACTION))
        high = dedupe(slice_[slice_.score >= threshold], horizon_minutes)
        r = high.net_r.to_numpy(dtype=float)
        if len(r) < MIN_TOP_EVENTS:
            evidence[name] = RegimeEvidence(name, threshold, n, len(r), None, None, False)
            continue
        mean = float(r.mean())
        uncertainty = UNCERTAINTY_MULTIPLE * float(r.std(ddof=1)) / math.sqrt(len(r))
        haircut = mean - uncertainty - ECONOMIC_BUFFER_R
        evidence[name] = RegimeEvidence(
            name, threshold, n, len(r), mean, float(haircut), bool(haircut > 0)
        )
    return evidence


def select_from_frozen_gates(
    test: pd.DataFrame, raw_predictions: np.ndarray,
    gates: dict[str, RegimeEvidence], *, side: str, horizon_minutes: int,
) -> pd.DataFrame:
    """Never inspect test outcomes to select candidates; return chronological rows."""
    if set(gates) != set(REGIMES) or not {"t", "symbol"}.issubset(test.columns):
        raise ValueError("missing complete four-regime model")
    scores = np.asarray(raw_predictions, dtype=float)
    if scores.shape != (len(test),) or not np.isfinite(scores).all():
        raise ValueError("invalid test predictions")
    test_copy = test.copy()
    test_copy["score"] = scores
    test_copy["regime"] = regime_of(test_copy, side=side)
    decisions = np.zeros(len(test_copy), dtype=bool)
    labels = test_copy["regime"].to_numpy()
    for name in REGIMES:
        evidence = gates[name]
        if evidence.regime != name:
            raise ValueError("mismatched regime evidence")
        if not evidence.eligible_for_research:
            continue
        if evidence.model_score_threshold is None or not math.isfinite(evidence.model_score_threshold):
            raise ValueError("eligible gate lacks threshold")
        decisions |= (labels == name) & (scores >= evidence.model_score_threshold)
    return dedupe(test_copy.loc[decisions], horizon_minutes)
