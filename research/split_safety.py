"""Independent walk-forward split validator; all boundaries are UTC-aware."""
from datetime import datetime, timedelta


def validate_windows(train_end: datetime, cal_start: datetime, cal_end: datetime,
                     test_start: datetime, label_horizon_minutes: int,
                     embargo_minutes: int) -> None:
    if any(x.tzinfo is None or x.utcoffset() is None for x in (train_end, cal_start, cal_end, test_start)):
        raise ValueError("timestamps must be timezone-aware")
    if label_horizon_minutes <= 0 or embargo_minutes < 0:
        raise ValueError("invalid horizon or embargo")
    if not train_end < cal_start < cal_end < test_start:
        raise ValueError("overlapping or reversed windows")
    gap = timedelta(minutes=label_horizon_minutes + embargo_minutes)
    if train_end + gap > cal_start:
        raise ValueError("train labels may overlap calibration")
    if cal_end + gap > test_start:
        raise ValueError("calibration labels may overlap test")
