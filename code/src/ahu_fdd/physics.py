"""Utility functions for the AHU fault detection assignment.

You may use, change or ignore any of these. They load the data, turn the label
table into per-row labels, and cover some basic air-side physics. They do NOT
clean the data for you.

Requires: numpy, pandas, pyarrow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FAULT_TYPES = ["chw_valve_stuck", "oa_damper_stuck", "sat_sensor_bias", "low_coil_capacity"]
RHO_CP_AIR = 1.2 * 1.006  # volumetric heat capacity of air, kJ/(m3 K)


# --------------------------------------------------------------------------- loading
def load_train() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "ahu_train.parquet")


def load_test() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "ahu_test.parquet")


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ahu_metadata.csv")


def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fault_labels_train.csv", parse_dates=["start", "end"])


def attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Per-row label from the event table: the fault type if the row falls in
    [start, end) of an event for that AHU, otherwise "normal"."""
    out = pd.Series("normal", index=df.index, dtype=object)
    for e in labels.itertuples(index=False):
        m = (df["ahu_id"] == e.ahu_id) & (df["timestamp"] >= e.start) & (df["timestamp"] < e.end)
        out[m] = e.fault_type
    return out


def running(df: pd.DataFrame) -> pd.Series:
    """Rows where the supply fan is on. Only these rows are scored."""
    return df["fan_status"] == 1


# --------------------------------------------------------------------------- psychrometrics
def saturation_pressure_kpa(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water (Magnus formula), kPa."""
    t = np.asarray(t_c, dtype=float)
    return 0.61094 * np.exp(17.625 * t / (t + 243.04))


def humidity_ratio(t_c, rh_pct, p_kpa: float = 101.325) -> np.ndarray:
    """kg water vapour per kg dry air."""
    pw = np.asarray(rh_pct, dtype=float) / 100 * saturation_pressure_kpa(t_c)
    return 0.62198 * pw / (p_kpa - pw)


def enthalpy_kj_per_kg(t_c, rh_pct) -> np.ndarray:
    """Moist air enthalpy, kJ per kg dry air."""
    w = humidity_ratio(t_c, rh_pct)
    t = np.asarray(t_c, dtype=float)
    return 1.006 * t + w * (2501 + 1.86 * t)


# --------------------------------------------------------------------------- air-side
def outdoor_air_fraction(mat_c, rat_c, oat_c, min_delta_c: float = 2.0) -> np.ndarray:
    """Estimate the outdoor-air fraction from a mixing energy balance:

        MAT = x * OAT + (1 - x) * RAT  ->  x = (MAT - RAT) / (OAT - RAT)

    Returns NaN when |OAT - RAT| < min_delta_c, where the estimate is unreliable
    (and it is noisy close to that limit too)."""
    mat, rat, oat = (np.asarray(v, dtype=float) for v in (mat_c, rat_c, oat_c))
    delta = oat - rat
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (mat - rat) / delta
    return np.where(np.abs(delta) >= min_delta_c, x, np.nan)


def sensible_cooling_kw(airflow_m3h, t_in_c, t_out_c) -> np.ndarray:
    """Sensible heat removed from an air stream, kW."""
    q = np.asarray(airflow_m3h, dtype=float) / 3600
    return RHO_CP_AIR * q * (np.asarray(t_in_c, dtype=float) - np.asarray(t_out_c, dtype=float))


def fan_affinity_power(speed_pct, rated_kw, exponent: float = 3.0) -> np.ndarray:
    """Ideal fan power at a given speed (affinity law). Real systems with
    static pressure control follow a lower exponent, typically 2 to 3."""
    return np.asarray(rated_kw, dtype=float) * (np.asarray(speed_pct, dtype=float) / 100) ** exponent


# --------------------------------------------------------------------------- helpers
def steady_state(series: pd.Series, window: int = 6, max_range: float = 0.5) -> pd.Series:
    """True where the rolling range (max - min) over `window` rows is within
    `max_range`. Useful for excluding start-up and transients. Apply per AHU."""
    r = series.rolling(window, min_periods=window)
    return (r.max() - r.min()) <= max_range


def per_ahu(df: pd.DataFrame, func, *cols, **kwargs) -> pd.Series:
    """Apply a series function per AHU, in time order, aligned to df's index.

    Example: per_ahu(df, steady_state, "sat_c", window=6)
    """
    df = df.sort_values(["ahu_id", "timestamp"])
    parts = [func(*(g[c] for c in cols), **kwargs) for _, g in df.groupby("ahu_id", sort=False)]
    return pd.concat(parts).reindex(df.index)
