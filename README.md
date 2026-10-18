# fishnet-data

Research dataset for Foxy Pro Fishnet: one year of minute bars for 60 USDT perpetuals, built from
Binance public archives (data.binance.vision) by `.github/workflows/collect.yml`.

`data/<SYMBOL>/<YYYY-MM>.parquet`, one row per minute (UTC):

- `open high low close vwap`: perp price
- `buy_q sell_q`: taker buy and sell notional (USDT); `n n_buy`: trade counts; `max_q`: largest trade
- `buy_s/m/l/w`, `sell_s/m/l/w`: notional by trade size bucket (below p50, p50 to p90, p90 to p99, above p99)
- `rush_q`: notional of trades stamped in the same millisecond as another (sweeps, bursts)
- `spot_close spot_buy_q spot_sell_q spot_n spot_max_q spot_rush_q`: the same coin on spot
  (for 1000x perps the spot price is per single coin)
- `premium`: perp premium over the index

`metrics-*.parquet`: 5-minute open interest, top trader and global long/short ratios, taker ratio.
`funding-*.parquet`: funding rate history. `cuts.json`: the size bucket thresholds, measured on
the first three days from 2025-10-15, which are not in the bars.
