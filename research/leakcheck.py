"""Leak sentinel: fit a quick model on four coins and print the strongest features and the
top-1% hit rate of each on a later month. A single feature near 0 or near 1 means look-ahead."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "research")
import run as R  # noqa: E402

df = pd.read_parquet(sys.argv[1])
df = df[df.symbol.isin(["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])]
cols = R.feature_cols(df)
X = df[cols].to_numpy(np.float32)
X[~np.isfinite(X)] = np.nan
t = pd.to_datetime(df.t, utc=True)
for yname in sys.argv[2:] or ["y_long_5"]:
    y = df[yname].to_numpy()
    m = ~np.isnan(y)
    tr = m & (t < pd.Timestamp("2026-04-01", tz="UTC")).to_numpy()
    te = m & ((t >= pd.Timestamp("2026-04-17", tz="UTC")) & (t < pd.Timestamp("2026-05-17", tz="UTC"))).to_numpy()
    b = lgb.train(dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.05), lgb.Dataset(X[tr], y[tr]), 200)
    p = b.predict(X[te])
    q = np.quantile(p, 0.99)
    print(f"{yname}: base {y[te].mean():.3f}, model top1% {y[te][p >= q].mean():.3f}")
    g = b.feature_importance("gain")
    for v, c in sorted(zip(g, cols), reverse=True)[:8]:
        x = X[te, cols.index(c)]
        lo, hi = np.nanquantile(x, [0.01, 0.99])
        print(f"  {c:22s} gain {v / g.sum():.3f}  low1% {np.nanmean(y[te][x <= lo]):.3f}  high1% {np.nanmean(y[te][x >= hi]):.3f}")
