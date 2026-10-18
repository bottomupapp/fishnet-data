"""Safe single-fold Fishnet development baseline. Never reads locked holdout labels.
This is diagnostic ML evaluation, not a deployable signal or profitability claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from research.run import HOLDOUT_START, EMBARGO, PARAMS, feature_cols, score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--target", default="y_long_15")
    ap.add_argument("--test-start", default="2026-07-17")
    args=ap.parse_args()
    a=pd.Timestamp(args.test_start, tz="UTC")
    end=min(a+pd.Timedelta(days=30), HOLDOUT_START-EMBARGO)
    if a>=end or a>=HOLDOUT_START-EMBARGO:
        raise ValueError("test period crosses locked holdout")
    df=pd.read_parquet(args.data)
    df["t"]=pd.to_datetime(df["t"],utc=True)
    df=df[df.t < HOLDOUT_START-EMBARGO].copy()
    if args.target not in df: raise ValueError("unknown target")
    purge=pd.Timedelta(minutes=60)
    cal_start=a-EMBARGO-pd.Timedelta(days=30)
    train=df[(df.t<cal_start-EMBARGO-purge)&df[args.target].notna()]
    cal=df[(df.t>=cal_start)&(df.t<a-EMBARGO-purge)&df[args.target].notna()]
    test=df[(df.t>=a)&(df.t<end)&df[args.target].notna()]
    if min(len(train),len(cal),len(test))==0: raise ValueError("empty train/cal/test partition")
    cols=feature_cols(df)
    dataset=lgb.Dataset(train[cols], train[args.target].astype(int),free_raw_data=True)
    model=lgb.train(dict(PARAMS),dataset,num_boost_round=args.rounds)
    y=args.target
    hz=int(y.rsplit("_",1)[-1])
    result=score(test,model.predict(test[cols]),y,hz,model.predict(cal[cols]))
    report={"target":y,"test_start":str(a),"test_end_exclusive":str(end),
            "holdout_locked_from":str(HOLDOUT_START),"rounds":args.rounds,
            "rows":{"train":len(train),"cal":len(cal),"test":len(test)},
            "features":len(cols),"metrics":result,
            "warning":"Research-only barrier hit rates; not returns or proof of profitability. No holdout evaluation."}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
