"""Check predictions.csv before you submit it.

    python starter/validate_submission.py path/to/predictions.csv

It checks the format only. It does not tell you how well the predictions score.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ALLOWED = {"normal", "chw_valve_stuck", "oa_damper_stuck", "sat_sensor_bias", "low_coil_capacity"}
# changed from the starter: same data location as the rest of the code (AHU_DATA_DIR, default ../data)
DATA = Path(os.environ.get("AHU_DATA_DIR", Path(__file__).resolve().parents[2] / "data")) / "ahu_test.parquet"


def main(path: str) -> int:
    pred = pd.read_csv(path)
    problems = []
    expected_cols = ["ahu_id", "timestamp", "fault_type"]
    if list(pred.columns[:3]) != expected_cols:
        problems.append(f"columns must start with {expected_cols}, got {list(pred.columns)}")
        return report(problems)

    pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="coerce")
    if pred["timestamp"].isna().any():
        problems.append(f"{pred['timestamp'].isna().sum()} timestamps could not be parsed")

    bad = sorted(set(pred["fault_type"].dropna().astype(str)) - ALLOWED)
    if bad:
        problems.append(f"unknown fault_type values: {bad[:10]} (allowed: {sorted(ALLOWED)})")
    if pred["fault_type"].isna().any():
        problems.append(f"{pred['fault_type'].isna().sum()} rows have an empty fault_type")

    dup = pred.duplicated(["ahu_id", "timestamp"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (ahu_id, timestamp) rows")

    test = pd.read_parquet(DATA, columns=["ahu_id", "timestamp"])
    keys = test.merge(pred[["ahu_id", "timestamp"]], on=["ahu_id", "timestamp"], how="left", indicator=True)
    missing = (keys["_merge"] == "left_only").sum()
    if missing:
        problems.append(f"{missing} rows of ahu_test.parquet have no prediction")
    extra = len(pred) - (len(test) - missing)
    if extra > 0:
        problems.append(f"{extra} predictions do not match any row of ahu_test.parquet")
    return report(problems, len(pred))


def report(problems, n=None) -> int:
    if problems:
        print("NOT VALID")
        for p in problems:
            print(" -", p)
        return 1
    print(f"OK: {n} rows, format is valid")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
