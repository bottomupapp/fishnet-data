"""Missing-minute guards for historical ML. No future data enters features."""
import numpy as np
import pandas as pd


def observed_mask(index: pd.DatetimeIndex, full: pd.DatetimeIndex) -> pd.Series:
    """True only for timestamps actually present in source files."""
    return pd.Series(full.isin(index), index=full, dtype=bool)


def valid_history(observed: pd.Series, minutes: int = 60) -> pd.Series:
    """A decision at t requires actual source bars throughout the lookback."""
    return observed.astype(int).rolling(minutes, min_periods=minutes).min().eq(1)


def valid_outcome(observed: pd.Series, horizon: int) -> pd.Series:
    """Labels at t require observed bars t+1 ... t+horizon."""
    future = observed.shift(-1, fill_value=False).astype(int).iloc[::-1]
    return future.rolling(horizon, min_periods=horizon).min().iloc[::-1].eq(1)


def mask_labels(labels: pd.DataFrame, observed: pd.Series, horizons=(5, 15, 60)) -> pd.DataFrame:
    """Set labels to NaN whenever future observations are incomplete."""
    result = labels.copy()
    for horizon in horizons:
        safe = valid_outcome(observed, horizon).to_numpy()
        for side in ("long", "short"):
            col = f"y_{side}_{horizon}"
            if col in result:
                result.loc[~safe, col] = np.nan
    return result
