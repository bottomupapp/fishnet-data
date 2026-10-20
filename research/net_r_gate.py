"""Pretest net-R calibration gate for Fishnet's risk-adjusted return pilot.

This gate works with REALIZED historical net-R outcomes from an earlier
calibration fold, not with theoretical barrier-hit frequencies. The 2-SE
haircut is a heuristic, NOT a statistically valid confidence interval for
autocorrelated market events. Only non-overlapping same-symbol events are
used in calibration. No live approval or order placement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from research.run import dedupe


@dataclass(frozen=True)
class NetRGate:
    score_threshold: float
    reference_events: int
    highest_decile_events: int
    highest_decile_realized_net_r: float | None
    haircut_net_r: float | None
    admitted_for_research: bool

    def select(self, predictions: np.ndarray) -> np.ndarray:
        a=np.asarray(predictions,dtype=float)
        if not np.isfinite(a).all():
            raise ValueError("invalid model predictions")
        return np.asarray(a >= self.score_threshold) & self.admitted_for_research


def calibrate_r_gate(
    cal: pd.DataFrame,
    cal_predictions: np.ndarray,
    *,
    horizon_minutes: int,
    upper_fraction: float = .1,
    minimum_high_scores: int = 30,
    economic_buffer_r: float = .05,
) -> NetRGate:
    """Predeclared top-decile threshold and conservative realized-R audit."""
    if not {"t","symbol","net_r"}.issubset(cal.columns):
        raise ValueError("calibration fold must include dated symbol net-R observations")
    if (not 1 <= horizon_minutes <= 240 or not 0 < upper_fraction < .5
        or type(minimum_high_scores) is not int or minimum_high_scores < 20
        or not math.isfinite(economic_buffer_r) or economic_buffer_r < 0):
        raise ValueError("invalid calibration policy")
    scores=np.asarray(cal_predictions,dtype=float)
    if scores.ndim != 1 or scores.shape[0] != len(cal) or not np.isfinite(scores).all():
        raise ValueError("invalid calibration predictions")
    values=cal["net_r"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("invalid historical realized net-R")
    events=cal[["t","symbol","net_r"]].copy()
    events["score"]=scores
    events=dedupe(events,horizon_minutes).copy()
    if len(events) < 200:
        raise ValueError("insufficient non-overlapping calibration events")
    threshold=float(np.quantile(events["score"].to_numpy(),1-upper_fraction))
    top=events[events["score"]>=threshold]["net_r"].to_numpy(dtype=float)
    if len(top) < minimum_high_scores:
        return NetRGate(threshold,len(events),len(top),None,None,False)
    mean=float(np.mean(top))
    sigma=float(np.std(top,ddof=1))
    # Portfolio autocorrelation and cross-asset linkage remain a limitation.
    heuristic_lower=mean-2*sigma/math.sqrt(len(top))-economic_buffer_r
    return NetRGate(
        score_threshold=threshold,
        reference_events=len(events),
        highest_decile_events=len(top),
        highest_decile_realized_net_r=mean,
        haircut_net_r=float(heuristic_lower),
        admitted_for_research=bool(heuristic_lower>0),
    )
