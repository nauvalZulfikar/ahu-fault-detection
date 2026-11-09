"""Paths and small shared loaders."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AHU_DATA_DIR", CODE_DIR.parent / "data"))
OUT_DIR = CODE_DIR / "outputs"

FAULTS = ["chw_valve_stuck", "oa_damper_stuck", "sat_sensor_bias", "low_coil_capacity"]
CLASSES = ["normal"] + FAULTS


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ahu_metadata.csv")


def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fault_labels_train.csv", parse_dates=["start", "end"])
