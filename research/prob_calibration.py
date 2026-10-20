"""Fishnet fail-closed monotone calibration for rare 3.2R target probabilities.

Fit ONLY on a historical calibration fold strictly earlier than the test
period. Monotone empirical-Bayes bins shrink rare hit rates toward their
observed base rate. This is an exploratory gate, not a guarantee that
probabilities generalize; locked holdout remains unopened.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Calibration:
    boundaries: tuple[float, ...]
    probabilities: tuple[float, ...]
    base_rate: float
    calibration_count: int
    calibration_positives: int

    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        values=np.asarray(raw_predictions, dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("raw model probabilities must be finite in [0,1]")
        buckets=np.searchsorted(self.boundaries,values,side="right")
        return np.asarray(self.probabilities,dtype=float)[buckets]


def fit_monotone_calibrator(
    historical_scores: np.ndarray,
    historical_labels: np.ndarray,
    *,
    max_bins: int = 10,
    prior_strength: float = 200.,
) -> Calibration:
    """Fit equal-mass quantile buckets with shrinkage and PAV monotonicity."""
    scores=np.asarray(historical_scores,dtype=float)
    labels=np.asarray(historical_labels,dtype=float)
    if scores.ndim != 1 or labels.shape != scores.shape or len(scores) < 200:
        raise ValueError("insufficient calibration sample")
    if (not np.isfinite(scores).all() or not np.isfinite(labels).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((labels != 0) & (labels != 1))):
        raise ValueError("invalid probabilities or calibration labels")
    if type(max_bins) is not int or not 2 <= max_bins <= 100:
        raise ValueError("invalid calibration bin count")
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("invalid calibration shrinkage")
    positive=int(labels.sum())
    if not 5 <= positive <= len(labels)-5:
        raise ValueError("too few calibration positive or negative observations")
    base=float(labels.mean())
    boundaries=np.unique(np.quantile(scores,np.linspace(0,1,max_bins+1)[1:-1]))
    bucket=np.searchsorted(boundaries,scores,side="right")
    k=len(boundaries)+1
    counts=np.bincount(bucket,minlength=k).astype(float)
    hits=np.bincount(bucket,weights=labels,minlength=k)
    # Empty buckets, if any, shrink exactly to the training base rate.
    weight=counts+prior_strength
    weighted_hits=hits+prior_strength*base

    blocks=[]
    for b in range(k):
        blocks.append([b,b,float(weighted_hits[b]),float(weight[b])])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            right=blocks.pop()
            blocks[-1][1]=right[1]
            blocks[-1][2]+=right[2]
            blocks[-1][3]+=right[3]
    calibrated=np.empty(k,dtype=float)
    for first,last,summed,denom in blocks:
        calibrated[first:last+1]=summed/denom
    if not (np.isfinite(calibrated).all() and np.all((calibrated>=0)&(calibrated<=1))):
        raise ValueError("invalid calibration fit")
    return Calibration(tuple(float(v) for v in boundaries),
                       tuple(float(v) for v in calibrated),
                       base,len(scores),positive)
