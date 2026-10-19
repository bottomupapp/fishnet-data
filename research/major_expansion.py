"""Fail-closed integrity and label coverage gate for Fishnet major-coin trials.

No holdout rows are used for model selection. Missing historical bars are
accounted for by research.features.mask_labels instead of forward-filled labels.
"""
from __future__ import annotations
import argparse
import json
import pandas as pd

MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
LABELS = ("y_long_5", "y_short_5", "y_long_15", "y_short_15", "y_long_60", "y_short_60")
HOLDOUT_START = pd.Timestamp("2026-08-17T00:00:00Z")


def audit_features(df: pd.DataFrame, required=MAJORS) -> dict:
    if not {"symbol", "t", *LABELS}.issubset(df.columns):
        raise ValueError("required feature/label columns missing")
    if tuple(sorted(df.symbol.unique())) != tuple(sorted(required)):
        raise ValueError("missing or unexpected major symbol")
    times = pd.to_datetime(df.t, utc=True)
    if df.duplicated(["symbol", "t"]).any():
        raise ValueError("duplicate coin-minute features")
    if (times >= HOLDOUT_START).all():
        raise ValueError("no development observations")
    counts={}
    for sym in required:
        subset=df[df.symbol == sym]
        if len(subset) < 30000:
            raise ValueError(f"insufficient feature history: {sym}")
        counts[sym]={"rows":int(len(subset)), "finite_label_counts":{}}
        for label in LABELS:
            nonnull=int(subset[label].notna().sum())
            if nonnull < 20000:
                raise ValueError(f"insufficient {label} coverage: {sym}")
            counts[sym]["finite_label_counts"][label]=nonnull
    return {"symbols":list(required), "coverage":counts, "training_universe_approved":False,
            "model_live_approved":False, "locked_holdout_evaluated":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    df=pd.read_parquet(args.features)
    result=audit_features(df)
    with open(args.output,"w") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
