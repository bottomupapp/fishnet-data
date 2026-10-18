import unittest
from datetime import datetime, timedelta, timezone
from research.split_safety import validate_windows

T = datetime(2026, 5, 1, tzinfo=timezone.utc)

class SplitSafetyTests(unittest.TestCase):
    def test_valid_gap(self):
        validate_windows(T, T+timedelta(days=2), T+timedelta(days=30), T+timedelta(days=32), 60, 1440)
    def test_leaking_training_label_rejected(self):
        with self.assertRaises(ValueError):
            validate_windows(T, T+timedelta(minutes=20), T+timedelta(days=5), T+timedelta(days=7), 60, 1440)
    def test_leaking_calibration_label_rejected(self):
        with self.assertRaises(ValueError):
            validate_windows(T, T+timedelta(days=2), T+timedelta(days=30), T+timedelta(days=30, minutes=5), 60, 1440)
    def test_naive_time_rejected(self):
        with self.assertRaises(ValueError):
            validate_windows(T.replace(tzinfo=None), T+timedelta(days=2), T+timedelta(days=30), T+timedelta(days=32), 60, 1440)
