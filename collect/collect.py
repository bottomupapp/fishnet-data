"""Fishnet research dataset: one symbol, one year of minute bars from Binance public archives.

Sources (data.binance.vision, free, no key):
  futures/um/daily/aggTrades      every aggregated perp trade with its taker side
  futures/um/daily/metrics        5-minute open interest, long/short ratios, taker ratio
  futures/um/daily/premiumIndexKlines/1m   perp premium over the index, per minute
  futures/um/monthly/fundingRate  funding rate history
  spot/daily/aggTrades            every aggregated spot trade (same coin, USDT pair)

Writes data/<SYMBOL>/<YYYY-MM>.parquet (one row per minute) plus metrics.parquet and funding.parquet.
Size buckets (small / medium / large / whale) use thresholds measured on the first three days,
which are not written, so no bar uses information from its own future.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
AGG = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
SPOT_AGG = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "best_match"]
SESSION = requests.Session()


def get_csv(url: str, names: list[str] | None = None) -> pd.DataFrame | None:
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    first = f.read(1)
                with z.open(name) as f:
                    if first.isdigit() and names:
                        return pd.read_csv(f, header=None, names=names)
                    df = pd.read_csv(f)
                    if names and len(df.columns) == len(names) and str(df.columns[0])[:1].isdigit():
                        f2 = z.open(name)
                        return pd.read_csv(f2, header=None, names=names)
                    return df
        except Exception as e:  # network hiccup: retry
            if attempt == 3:
                print(f"giving up on {url}: {e}", flush=True)
                return None
    return None


def trades_to_minutes(df: pd.DataFrame, cuts: np.ndarray | None, prefix: str) -> pd.DataFrame:
    """Minute bars from aggregated trades. cuts = notional thresholds for the size buckets."""
    df.columns = [str(c).strip() for c in df.columns]
    px = df["price"].astype(np.float64).to_numpy()
    q = px * df["quantity"].astype(np.float64).to_numpy()
    maker = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    ts = df["transact_time"].astype(np.int64).to_numpy()
    if ts.max() > 10**14:  # spot archives switched to microseconds
        ts = ts // 1000
    minute = (ts // 60_000) * 60_000
    buy = np.where(~maker, q, 0.0)
    sell = np.where(maker, q, 0.0)
    cols = {"m": minute, "px": px, "buy": buy, "sell": sell, "n": np.ones_like(q), "nbuy": (~maker).astype(float), "qmax": q, "pq": px * q}
    # Same-millisecond bursts: several aggregated trades stamped the same ms (rush orders, sweeps).
    same_ms = pd.Series(ts).duplicated(keep=False).to_numpy()
    cols["rush_q"] = np.where(same_ms, q, 0.0)
    if cuts is not None:
        b = np.searchsorted(cuts, q, side="right")  # 0 small, 1 medium, 2 large, 3 whale
        for k, name in enumerate(["s", "m", "l", "w"]):
            cols[f"buy_{name}"] = np.where(b == k, buy, 0.0)
            cols[f"sell_{name}"] = np.where(b == k, sell, 0.0)
    g = pd.DataFrame(cols).groupby("m", sort=True)
    agg = {"px": ["first", "max", "min", "last"], "buy": "sum", "sell": "sum", "n": "sum", "nbuy": "sum", "qmax": "max", "pq": "sum", "rush_q": "sum"}
    if cuts is not None:
        for name in ["s", "m", "l", "w"]:
            agg[f"buy_{name}"] = "sum"
            agg[f"sell_{name}"] = "sum"
    out = g.agg(agg)
    out.columns = ["open", "high", "low", "close", "buy_q", "sell_q", "n", "n_buy", "max_q", "pq", "rush_q"] + (
        [f"{s}_{n}" for n in ["s", "m", "l", "w"] for s in ["buy", "sell"]] if cuts is not None else []
    )
    vol = out["buy_q"] + out["sell_q"]
    out["vwap"] = out["pq"] / vol.where(vol > 0)
    out = out.drop(columns=["pq"])
    out.index = pd.to_datetime(out.index, unit="ms", utc=True)
    out.index.name = "t"
    return out.add_prefix(prefix)


def day(symbol: str, spot: str | None, d: date, cuts: np.ndarray | None):
    ds = d.isoformat()
    fut = get_csv(f"{BASE}/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{ds}.zip", AGG)
    if fut is None:
        return ds, None, None, None, None
    raw_q = None
    if cuts is None:
        raw_q = fut["price"].astype(float).to_numpy() * fut["quantity"].astype(float).to_numpy()
        return ds, None, None, None, raw_q
    bars = trades_to_minutes(fut, cuts, "")
    if spot:
        sp = get_csv(f"{BASE}/spot/daily/aggTrades/{spot}/{spot}-aggTrades-{ds}.zip", SPOT_AGG)
        if sp is not None and len(sp):
            s = trades_to_minutes(sp, None, "spot_")[["spot_close", "spot_buy_q", "spot_sell_q", "spot_n", "spot_max_q", "spot_rush_q"]]
            bars = bars.join(s, how="left")
    prem = get_csv(f"{BASE}/futures/um/daily/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{ds}.zip",
                   ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "count", "tbv", "tbqv", "ignore"])
    if prem is not None and len(prem):
        p = prem.copy()
        p.columns = [str(c).strip() for c in p.columns]
        p["t"] = pd.to_datetime(p["open_time"].astype(np.int64), unit="ms", utc=True)
        bars = bars.join(p.set_index("t")["close"].astype(float).rename("premium"), how="left")
    met = get_csv(f"{BASE}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{ds}.zip")
    return ds, bars, met, None, None


def main() -> None:
    symbol, start, end, out_root = sys.argv[1], date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3]), sys.argv[4]
    spot = symbol[4:] if symbol.startswith("1000") else symbol
    if os.environ.get("NO_SPOT") == "1":
        spot = None
    part = os.environ.get("PART", "all")
    out = os.path.join(out_root, symbol)
    os.makedirs(out, exist_ok=True)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # Size thresholds from the first three days with data from CUTS_START (default: start).
    # Every part of a symbol uses the same window, so buckets mean the same thing all year.
    cs = date.fromisoformat(os.environ.get("CUTS_START", sys.argv[2]))
    sample, k = [], 0
    while len(sample) < 3 and k < 30:
        _, _, _, _, q = day(symbol, None, cs + timedelta(days=k), None)
        if q is not None:
            sample.append(q)
        k += 1
    cut_days = {cs + timedelta(days=j) for j in range(k)}
    i = 0
    days = [d for d in days if d not in cut_days]
    if not sample:
        print(f"{symbol}: no data")
        return
    allq = np.concatenate(sample)
    cuts = np.quantile(allq, [0.5, 0.9, 0.99])
    print(f"{symbol}: size cuts p50 {cuts[0]:,.0f}  p90 {cuts[1]:,.0f}  p99 {cuts[2]:,.0f} USDT", flush=True)
    pd.Series(cuts, index=["p50", "p90", "p99"]).to_json(os.path.join(out, "cuts.json"))

    rest = days[i:]
    months: dict[str, list[pd.DataFrame]] = {}
    metrics = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for ds, bars, met, _, _ in ex.map(lambda d: day(symbol, spot, d, cuts), rest):
            if bars is None:
                print(f"{symbol} {ds}: missing", flush=True)
                continue
            bars = bars.astype(np.float32)
            months.setdefault(ds[:7], []).append(bars)
            if met is not None:
                metrics.append(met)
            print(f"{symbol} {ds}: {len(bars)} min", flush=True)
    for ym, parts in months.items():
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(os.path.join(out, f"{ym}.parquet"), compression="zstd")
    if metrics:
        m = pd.concat(metrics)
        m.columns = [str(c).strip() for c in m.columns]
        m["t"] = pd.to_datetime(m["create_time"], utc=True)
        keep = [c for c in m.columns if c not in ("create_time", "symbol")]
        m = m[keep].drop_duplicates("t").sort_values("t")
        for c in m.columns:
            if c != "t":
                m[c] = pd.to_numeric(m[c], errors="coerce").astype(np.float32)
        m.to_parquet(os.path.join(out, f"metrics-{part}.parquet"), compression="zstd", index=False)
    # Funding history, monthly files.
    fund = []
    ym = date(start.year, start.month, 1)
    while ym <= end:
        f = get_csv(f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym:%Y-%m}.zip")
        if f is not None:
            fund.append(f)
        ym = date(ym.year + (ym.month == 12), ym.month % 12 + 1, 1)
    if fund:
        f = pd.concat(fund)
        f.columns = [str(c).strip() for c in f.columns]
        f.to_parquet(os.path.join(out, f"funding-{part}.parquet"), compression="zstd", index=False)
    print(f"{symbol}: {sum(len(v) for v in months.values())} days written", flush=True)


if __name__ == "__main__":
    main()
