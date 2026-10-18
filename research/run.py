"""Fishnet research loop: walk-forward evaluation of one model configuration.

Usage:
  python research/run.py <features.parquet> <iteration-name> [--holdout]

Data after HOLDOUT_START is locked. Without --holdout it is dropped before anything else, so
an iteration cannot look at it by accident. --holdout trains on everything before the lock
and scores the locked months once; every such run is appended to research/holdout_log.jsonl.

The question each iteration answers: is there a slice of moments where the model's top
predictions win at least 70% of the time, every month, often enough to be a product?
"""

from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2026-08-17", tz="UTC")
FOLDS = ["2026-04-17", "2026-05-17", "2026-06-17", "2026-07-17", "2026-08-17"]  # test windows between these
EMBARGO = pd.Timedelta("1D")
TOP = (0.001, 0.0025, 0.005, 0.01, 0.02)
FLOOR = 0.70
MIN_PER_MONTH = 60  # fewer winners than two a day across all coins is not a product
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=7)
ROUNDS = 400


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"t", "symbol", "sigma_1d"}
    return [c for c in df.columns if c not in drop and not c.startswith(("y_", "barrier_"))]


def dedupe(sel: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """One signal per symbol per outcome window: the alert would not fire again while it runs."""
    keep = []
    for _, g in sel.sort_values("t").groupby("symbol"):
        last = None
        for i, t in zip(g.index, g["t"]):
            if last is None or (t - last) >= pd.Timedelta(minutes=minutes):
                keep.append(i)
                last = t
    return sel.loc[keep]


def score(test: pd.DataFrame, p: np.ndarray, y: str, hz: int, cal: np.ndarray) -> dict:
    """Thresholds come from the calibration month before the test window, never from the test."""
    d = test[["t", "symbol", y]].copy()
    d["p"] = p
    d = d[d[y].notna()]
    res = {"n": int(len(d)), "base": float(d[y].mean())}
    for q in TOP:
        thr = float(np.quantile(cal, 1 - q))
        s = dedupe(d[d["p"] >= thr], hz)
        res[f"top{q}"] = {"thr": float(thr), "n": int(len(s)), "hit": float(s[y].mean()) if len(s) else None}
    return res


def fit(train: pd.DataFrame, cols: list[str], y: str) -> lgb.Booster:
    tr = train[train[y].notna()]
    ds = lgb.Dataset(tr[cols], tr[y], free_raw_data=True)
    return lgb.train(PARAMS, ds, ROUNDS)


def main() -> None:
    path, name = sys.argv[1], sys.argv[2]
    holdout = "--holdout" in sys.argv
    df = pd.read_parquet(path)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    if not holdout:
        df = df[df["t"] < HOLDOUT_START - EMBARGO]
    cols = feature_cols(df)
    hz_list = sorted({int(c.split("_")[-1]) for c in df.columns if c.startswith("y_long_")})
    out = {"name": name, "rows": int(len(df)), "features": len(cols), "folds": [], "importance": {}}
    t0 = time.time()
    windows = [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in zip(FOLDS[:-1], FOLDS[1:])]
    if holdout:
        windows = [(HOLDOUT_START, df["t"].max() + pd.Timedelta("1min"))]
    for a, b in windows:
        # A label at t can depend on prices through t + max(horizon).
        # Exclude boundary-crossing labels from train/calibration, not just test.
        label_purge = pd.Timedelta(minutes=max(hz_list))
        cal_start = a - EMBARGO - pd.Timedelta("30D")
        train = df[df["t"] < cal_start - EMBARGO - label_purge]
        cal = df[(df["t"] >= cal_start) & (df["t"] < a - EMBARGO - label_purge)]
        test = df[(df["t"] >= a) & (df["t"] < b)]
        fold = {"test": f"{a.date()}..{b.date()}", "train_rows": int(len(train)), "res": {}}
        for hz in hz_list:
            for side in ("long", "short"):
                y = f"y_{side}_{hz}"
                m = fit(train, cols, y)
                fold["res"][y] = score(test, m.predict(test[cols]), y, hz, m.predict(cal[cols]))
                imp = dict(zip(cols, m.feature_importance("gain")))
                acc = out["importance"].setdefault(y, {})
                for k, v in imp.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                print(f"{fold['test']} {y}: base {fold['res'][y]['base']:.3f} "
                      + " ".join(f"{q}:{fold['res'][y][f'top{q}']['hit'] or 0:.2f}/{fold['res'][y][f'top{q}']['n']}" for q in TOP)
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
        out["folds"].append(fold)
    for y, imp in out["importance"].items():
        tot = sum(imp.values()) or 1.0
        out["importance"][y] = dict(sorted(((k, round(v / tot, 4)) for k, v in imp.items()), key=lambda kv: -kv[1])[:20])
    # Verdict: a (target, slice) passes when every fold clears the floor with enough signals.
    verdict = []
    for y in out["folds"][0]["res"]:
        for q in TOP:
            hits = [f["res"][y][f"top{q}"]["hit"] or 0.0 for f in out["folds"]]
            ns = [f["res"][y][f"top{q}"]["n"] for f in out["folds"]]
            verdict.append({"target": y, "slice": q, "min_hit": round(min(hits), 3), "mean_hit": round(float(np.mean(hits)), 3),
                            "min_n": int(min(ns)), "pass": min(hits) >= FLOOR and min(ns) >= MIN_PER_MONTH})
    out["verdict"] = sorted(verdict, key=lambda v: -v["min_hit"])
    out["seconds"] = round(time.time() - t0)
    os.makedirs("research/iterations", exist_ok=True)
    tag = "holdout" if holdout else "dev"
    with open(f"research/iterations/{name}-{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    if holdout:
        with open("research/holdout_log.jsonl", "a") as fh:
            fh.write(json.dumps({"name": name, "at": pd.Timestamp.now(tz="UTC").isoformat(), "best": out["verdict"][:3]}) + "\n")
    print(json.dumps(out["verdict"][:8], indent=1))


if __name__ == "__main__":
    main()
