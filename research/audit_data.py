"""Read-only quality audit for monthly Fishnet parquet market bars.

Usage:
  python research/audit_data.py data BTCUSDT ETHUSDT

Does not rewrite observations or infer missing prices. Exits nonzero if
timestamp, OHLC, or duplicate checks fail; reports missing minute gaps.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import pandas as pd
import numpy as np


def audit(root: str, symbol: str) -> dict:
    paths = sorted(glob.glob(os.path.join(root, symbol, "20??-??.parquet")))
    report = {"symbol": symbol, "files": len(paths), "rows": 0, "duplicate_minutes": 0,
              "missing_minutes": None, "invalid_ohlc": 0, "nonpositive_close": 0,
              "invalid_volume": 0, "first": None, "last": None, "passed": False}
    if not paths:
        report["error"] = "no_monthly_files"
        return report
    parts = [pd.read_parquet(p) for p in paths]
    df = pd.concat(parts).sort_index()
    idx = pd.to_datetime(df.index, utc=True)
    report["rows"] = len(df)
    report["duplicate_minutes"] = int(idx.duplicated().sum())
    report["first"], report["last"] = idx.min().isoformat(), idx.max().isoformat()
    expected = len(pd.date_range(idx.min(), idx.max(), freq="1min", tz="UTC"))
    report["missing_minutes"] = int(expected - idx.nunique())
    numeric = {name: pd.to_numeric(df[name], errors="coerce") for name in ("open", "high", "low", "close", "buy_q", "sell_q")}
    report["invalid_ohlc"] = int((~np.isfinite(numeric["open"]) | ~np.isfinite(numeric["high"]) |
        ~np.isfinite(numeric["low"]) | ~np.isfinite(numeric["close"]) |
        (numeric["high"] < numeric["low"]) |
        (numeric["high"] < numeric["open"]) | (numeric["high"] < numeric["close"]) |
        (numeric["low"] > numeric["open"]) | (numeric["low"] > numeric["close"])).sum())
    report["nonpositive_close"] = int((numeric["close"] <= 0).sum())
    report["invalid_volume"] = int(((numeric["buy_q"] < 0) | (numeric["sell_q"] < 0) |
        ~np.isfinite(numeric["buy_q"]) | ~np.isfinite(numeric["sell_q"])).sum())
    report["passed"] = all(report[k] == 0 for k in
        ("duplicate_minutes", "invalid_ohlc", "nonpositive_close", "invalid_volume"))
    return report


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    symbols = sys.argv[2:] or ["BTCUSDT", "ETHUSDT"]
    results = [audit(root, symbol) for symbol in symbols]
    print(json.dumps(results, indent=2))
    # Gaps are reported separately; sparse minutes may be valid in low-liquidity markets.
    sys.exit(0 if all(result["passed"] for result in results) else 1)
