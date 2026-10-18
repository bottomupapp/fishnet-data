"""Fishnet research: causal features and triple-barrier labels from the minute dataset.

Every feature at minute t uses bars up to and including t. The trade is entered at the open of
t+1, so nothing the model sees overlaps the outcome window.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

try:
    from research.quality import valid_history, mask_labels
except ModuleNotFoundError:
    from quality import valid_history, mask_labels

HORIZONS = (5, 15, 60)
FEE_FLOOR = 0.002  # 0.2%: round-trip taker fees plus spread, a barrier must clear it
BARRIER_K = 1.0  # barrier = max(floor, k * sigma_1m * sqrt(h)), sigma over the prior day


def load_symbol(root: str, sym: str) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    files = sorted(f for f in glob.glob(os.path.join(root, sym, "*.parquet")) if os.path.basename(f)[:2] == "20")
    bars = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    full = pd.date_range(bars.index[0], bars.index[-1], freq="1min", tz="UTC")
    observed = pd.Series(full.isin(bars.index), index=full, dtype=bool)
    bars = bars.reindex(full)
    bars["source_observed"] = observed
    flows = [c for c in bars.columns if c.endswith("_q") or c.startswith(("buy_", "sell_")) or c in ("n", "n_buy", "spot_n")]
    bars[flows] = bars[flows].fillna(0.0)
    bars["close"] = bars["close"].ffill()
    for c in ("open", "high", "low", "vwap"):
        bars[c] = bars[c].fillna(bars["close"])
    for c in ("spot_close", "premium"):
        if c in bars:
            bars[c] = bars[c].ffill(limit=60)
    met = [pd.read_parquet(f) for f in glob.glob(os.path.join(root, sym, "metrics-*.parquet"))]
    met = pd.concat(met).drop_duplicates("t").set_index("t").sort_index() if met else None
    fund = [pd.read_parquet(f) for f in glob.glob(os.path.join(root, sym, "funding-*.parquet"))]
    fund = pd.concat(fund) if fund else None
    return bars.astype(np.float32), met, fund


def _sum(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=1).sum()


def _imb(b: pd.Series, s: pd.Series, w: int) -> pd.Series:
    bb, ss = _sum(b, w), _sum(s, w)
    return (bb - ss) / (bb + ss).where(bb + ss > 0)


def _z(x: pd.Series, w: int) -> pd.Series:
    m = x.rolling(w, min_periods=w // 4).mean()
    s = x.rolling(w, min_periods=w // 4).std()
    return (x - m) / s.where(s > 0)


def features(bars: pd.DataFrame, met: pd.DataFrame | None, fund: pd.DataFrame | None) -> pd.DataFrame:
    c = bars["close"].astype(np.float64)
    lc = np.log(c)
    r1 = lc.diff()
    vol = bars["buy_q"] + bars["sell_q"]
    f = {}
    for w in (1, 5, 15, 60, 240, 1440):
        f[f"ret_{w}"] = lc - lc.shift(w)
    sig = r1.rolling(1440, min_periods=360).std()
    f["sigma_1d"] = sig
    for w in (15, 60, 240):
        f[f"rv_{w}"] = r1.rolling(w, min_periods=w // 2).std() / sig
    for w in (5, 15, 60):
        f[f"ret_{w}_n"] = f[f"ret_{w}"] / (sig * np.sqrt(w))
    for w in (1, 5, 15, 60, 240):
        f[f"imb_{w}"] = _imb(bars["buy_q"], bars["sell_q"], w)
    # Volume and signed flow against the past week (10,080 minutes), on 15-minute sums.
    v15 = np.log1p(_sum(vol, 15))
    f["vol15_z"] = _z(v15, 10080)
    f["vol60_z"] = _z(np.log1p(_sum(vol, 60)), 10080)
    net15 = _sum(bars["buy_q"] - bars["sell_q"], 15)
    f["net15_z"] = net15 / _sum(vol, 15).rolling(10080, min_periods=2000).median()
    f["n15_z"] = _z(np.log1p(_sum(bars["n"], 15)), 10080)
    f["avg_trade_15"] = np.log1p(_sum(vol, 15) / _sum(bars["n"], 15).where(_sum(bars["n"], 15) > 0))
    for k in ("w", "l"):
        for w in (15, 60):
            f[f"imb_{k}_{w}"] = _imb(bars[f"buy_{k}"], bars[f"sell_{k}"], w)
        f[f"share_{k}_60"] = (_sum(bars[f"buy_{k}"] + bars[f"sell_{k}"], 60) / _sum(vol, 60).where(_sum(vol, 60) > 0))
    f["imb_s_60"] = _imb(bars["buy_s"], bars["sell_s"], 60)
    # Large trades buying while small trades sell, and the reverse.
    f["big_vs_small_60"] = f["imb_l_60"] - f["imb_s_60"]
    f["rush_share_15"] = _sum(bars["rush_q"], 15) / _sum(vol, 15).where(_sum(vol, 15) > 0)
    f["max_q_15_rel"] = np.log1p(bars["max_q"].rolling(15, min_periods=1).max()) - np.log1p(_sum(vol, 15) / 15)
    if "spot_buy_q" in bars:
        svol = bars["spot_buy_q"] + bars["spot_sell_q"]
        for w in (15, 60):
            f[f"spot_imb_{w}"] = _imb(bars["spot_buy_q"], bars["spot_sell_q"], w)
        f["spot_share_60"] = _sum(svol, 60) / (_sum(svol, 60) + _sum(vol, 60)).where(_sum(vol, 60) > 0)
        f["spot_perp_imb_gap_60"] = f["spot_imb_60"] - f["imb_60"]
        basis = np.log(bars["spot_close"].astype(np.float64)) - lc
        f["basis_dev"] = basis - basis.rolling(1440, min_periods=360).median()
    if "premium" in bars:
        p = bars["premium"].astype(np.float64)
        f["premium"] = p
        f["premium_chg_60"] = p - p.shift(60)
    hi, lo = c.rolling(240, min_periods=60).max(), c.rolling(240, min_periods=60).min()
    f["pos_240"] = (c - lo) / (hi - lo).where(hi > lo)
    hi1, lo1 = c.rolling(1440, min_periods=360).max(), c.rolling(1440, min_periods=360).min()
    f["dist_hi_1d"] = (np.log(hi1) - lc) / (sig * np.sqrt(60))
    f["dist_lo_1d"] = (lc - np.log(lo1)) / (sig * np.sqrt(60))
    vw = (bars["vwap"].astype(np.float64) * vol).rolling(240, min_periods=60).sum() / vol.rolling(240, min_periods=60).sum()
    f["vwap_dist_240"] = (lc - np.log(vw)) / (sig * np.sqrt(60))
    out = pd.DataFrame(f, index=bars.index)
    if met is not None and len(met):
        m = met.copy()
        # A 5-minute row stamped T describes the period ending at T; it becomes known at T.
        m = m.reindex(bars.index, method="ffill", limit=10)
        oi = np.log(m["sum_open_interest_value"].astype(np.float64)) if "sum_open_interest_value" in m else None
        if oi is not None:
            for w in (5, 60, 240, 1440):
                out[f"oi_chg_{w}"] = oi - oi.shift(w)
            out["oi_chg_60_z"] = _z(out["oi_chg_60"], 10080)
            # Price up with OI up = new longs; price up with OI down = short covering, etc.
            out["oi_x_ret_60"] = np.sign(out["ret_60"]) * out["oi_chg_60"]
        for col, name in (("count_toptrader_long_short_ratio", "ls_top"), ("sum_toptrader_long_short_ratio", "ls_top_pos"),
                          ("count_long_short_ratio", "ls_all"), ("sum_taker_long_short_vol_ratio", "taker_ratio")):
            if col in m:
                x = np.log(m[col].astype(np.float64).where(m[col] > 0))
                out[name] = x
                out[f"{name}_chg_240"] = x - x.shift(240)
                out[f"{name}_z"] = _z(x, 10080)
    if fund is not None and len(fund):
        fr = fund.copy()
        tcol = "calc_time" if "calc_time" in fr else fr.columns[0]
        fr["t"] = pd.to_datetime(fr[tcol].astype(np.int64), unit="ms", utc=True)
        rate = fr.set_index("t").sort_index()["last_funding_rate"].astype(np.float64)
        rate = rate[~rate.index.duplicated()]
        out["funding"] = rate.reindex(bars.index, method="ffill")
    out["hour"] = bars.index.hour
    out["dow"] = bars.index.dayofweek
    return out


def labels(bars: pd.DataFrame, sig: pd.Series) -> pd.DataFrame:
    """Long and short outcomes per horizon: 1 if the favourable barrier is hit first."""
    o = bars["open"].to_numpy(np.float64)
    h = bars["high"].to_numpy(np.float64)
    l = bars["low"].to_numpy(np.float64)
    n = len(o)
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    s = sig.to_numpy(np.float64)
    out = {}
    for hz in HORIZONS:
        b = np.maximum(FEE_FLOOR, BARRIER_K * s * np.sqrt(hz))
        up_px, dn_px = entry * (1 + b), entry * (1 - b)
        first_up = np.full(n, np.inf)
        first_dn = np.full(n, np.inf)
        for k in range(1, hz + 1):
            hk = np.full(n, np.nan)
            lk = np.full(n, np.nan)
            hk[:-k] = h[k:]
            lk[:-k] = l[k:]
            first_up = np.where(np.isinf(first_up) & (hk >= up_px), k, first_up)
            first_dn = np.where(np.isinf(first_dn) & (lk <= dn_px), k, first_dn)
        valid = ~np.isnan(entry) & ~np.isnan(s)
        valid[n - hz - 1:] = False
        # Both barriers in the same minute: the order is unknown, so count it as a loss both ways.
        long_win = (first_up < first_dn)
        short_win = (first_dn < first_up)
        out[f"y_long_{hz}"] = np.where(valid, long_win, np.nan).astype(np.float32)
        out[f"y_short_{hz}"] = np.where(valid, short_win, np.nan).astype(np.float32)
        out[f"barrier_{hz}"] = b.astype(np.float32)
    return pd.DataFrame(out, index=bars.index)


def build(root: str, symbols: list[str], stride: int = 5) -> pd.DataFrame:
    """Sample every `stride` minutes across symbols, add market-wide context."""
    parts = []
    for sym in symbols:
        try:
            bars, met, fund = load_symbol(root, sym)
        except ValueError:
            continue
        f = features(bars, met, fund)
        observed = bars["source_observed"].astype(bool)
        y = mask_labels(labels(bars, f["sigma_1d"]), observed, HORIZONS)
        f = f.where(valid_history(observed, 1440), other=np.nan)
        d = pd.concat([f, y], axis=1).iloc[::stride]
        d = d[d["sigma_1d"].notna()].astype(np.float32)
        d["symbol"] = sym
        parts.append(d)
        print(f"{sym}: {len(d):,} rows", flush=True)
    df = pd.concat(parts)
    df.index.name = "t"
    df = df.reset_index()
    # Market context at the same minute: BTC, and the cross-section of all coins.
    btc = df[df.symbol == "BTCUSDT"].set_index("t")[["ret_15", "ret_60", "ret_240", "imb_15", "imb_60", "oi_chg_60"]]
    df = df.join(btc.add_prefix("btc_"), on="t")
    g = df.groupby("t")
    for c in ("ret_60", "ret_240", "imb_60", "vol15_z", "oi_chg_60"):
        if c in df:
            df[f"xs_{c}"] = df[c] - g[c].transform("mean")
            df[f"mkt_{c}"] = g[c].transform("mean")
    df["xs_rank_ret_60"] = g["ret_60"].rank(pct=True)
    return df


def symbols(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


if __name__ == "__main__":
    import sys

    root, out = sys.argv[1], sys.argv[2]
    df = build(root, symbols(root))
    df.to_parquet(out, compression="zstd")
    print(json.dumps({"rows": len(df), "cols": df.shape[1]}))
