"""Load the raw BMS trend log and fix the data quality problems found in the EDA.

Every fix is logged so the report can quote exactly what was changed.
Nothing here uses fault labels: the same function runs on train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR

SIGNALS = [
    "fan_speed_pct", "fan_power_kw", "supply_static_pa", "supply_static_sp_pa", "sat_c", "sat_sp_c",
    "mat_c", "rat_c", "zone_temp_sp_c", "oat_c", "oa_rh_pct", "chw_valve_cmd_pct", "oa_damper_cmd_pct",
    "chwst_c", "filter_dp_pa", "co2_ppm", "sa_flow_m3h",
]
FLATLINE_ROWS = 36  # 3 h of identical readings while running = frozen sensor
FLATLINE_COLS = ["sat_c", "mat_c", "rat_c", "co2_ppm", "sa_flow_m3h", "filter_dp_pa"]


def load_raw(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"ahu_{split}.parquet")


def load_all_raw() -> pd.DataFrame:
    return pd.concat([load_raw("train").assign(split="train"), load_raw("test").assign(split="test")],
                     ignore_index=True)


def fix_clock_shift(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Undo a period where a building's logger wrote UTC instead of local time (UTC+8).

    Signature: every AHU of the building gets a block of duplicate timestamps (two different
    readings under one label) and, later, a gap of the same length. Rows labelled between the
    start of the duplicate block and the gap are moved forward by the block length.
    Inside the duplicate block the file holds the original reading first and the shifted one
    second. Checked against the other buildings' outdoor air temperature (same weather): the
    first copy matches them at the label time (mean error 0.35 C vs 3.4 C), the second copy
    matches them 8 h later (0.36 C vs 2.9 C).
    Exact re-exports (identical duplicate rows) are left for drop_exact_duplicates.
    """
    key = ["ahu_id", "timestamp"]
    out = []
    for b, g in df.groupby(df["ahu_id"].str[:2], sort=False):
        dup = g[g.duplicated(key, keep=False)]
        identical = dup.drop(columns="split").groupby(key).nunique(dropna=False).max(axis=1).le(1).mean()
        if dup.empty or identical > 0.9:
            out.append(g)
            continue
        start = dup["timestamp"].min()
        span = dup["timestamp"].max() - start + pd.Timedelta("5min")
        t = pd.Series(np.sort(g["timestamp"].unique()))
        after_gap = t[(t.diff() >= span) & (t > start)]
        if after_gap.empty:
            out.append(g)
            continue
        end = after_gap.iloc[0] - span
        g = g.copy()
        in_block = g["timestamp"].between(start, end, inclusive="left")
        overlap = in_block & g["timestamp"].lt(start + span)
        second_copy = g.duplicated(key, keep="first")  # relies on the file's row order
        shift = (in_block & ~overlap) | (overlap & second_copy)
        g.loc[shift, "timestamp"] += span
        log.append(dict(issue="clock shift (logger wrote UTC)", building=b, rows=int(shift.sum()),
                        detail=f"labels {start} to {end} moved forward by {span}"))
        out.append(g)
    return pd.concat(out)


def drop_exact_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    n0 = len(df)
    dup = df[df.duplicated(["ahu_id", "timestamp"], keep=False)]
    df = df.drop_duplicates(["ahu_id", "timestamp"], keep="first")
    if n0 != len(df):
        days = sorted(dup["timestamp"].dt.date.astype(str).unique())
        log.append(dict(issue="duplicate export", building=",".join(sorted(dup["ahu_id"].str[:2].unique())),
                        rows=n0 - len(df), detail=f"days {days}; kept first copy"))
    return df


def fix_power_units(df: pd.DataFrame, meta: pd.DataFrame, log: list) -> pd.DataFrame:
    """fan_power_kw far above the fan rating = logged in W instead of kW. Whole days from the first
    to the last such reading are converted, including fan-off standby power (40 W, not 40 kW)."""
    rated = df["ahu_id"].map(meta.set_index("ahu_id")["fan_rated_kw"])
    for a, g in df[df["fan_power_kw"] > 5 * rated].groupby("ahu_id"):
        lo, hi = g.timestamp.min().floor("D"), g.timestamp.max().ceil("D")
        bad = (df["ahu_id"] == a) & (df["timestamp"] >= lo) & (df["timestamp"] < hi)
        log.append(dict(issue="fan power in W, not kW", building=a, rows=int(bad.sum()),
                        detail=f"{lo.date()} to {(hi - pd.Timedelta('1D')).date()}: divided by 1000"))
        df.loc[bad, "fan_power_kw"] = df.loc[bad, "fan_power_kw"] / 1000
    return df


def mask_out_of_range(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """CO2 at 0 or at the 5000 ppm transmitter limit is a failed sensor, not air."""
    bad = (df["co2_ppm"] <= 0) | (df["co2_ppm"] >= 5000)
    if bad.any():
        for a, g in df[bad].groupby("ahu_id"):
            log.append(dict(issue="CO2 at 0 or 5000 ppm (sensor failure)", building=a, rows=len(g),
                            detail=f"{g.timestamp.min()} to {g.timestamp.max()}: set to NaN"))
        df.loc[bad, "co2_ppm"] = np.nan
    return df


def mask_flatlines(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """A reading that does not change at all for 3 h while the fan runs is a frozen point."""
    run = df["fan_status"] == 1
    for c in FLATLINE_COLS:
        x = df.loc[run, ["ahu_id", c]]
        same = x.groupby("ahu_id")[c].diff().eq(0)
        block = (~same).cumsum()
        length = same.groupby([x["ahu_id"], block]).transform("sum")
        frozen = length >= FLATLINE_ROWS
        if frozen.any():
            idx = frozen[frozen].index
            for a, g in df.loc[idx].groupby("ahu_id"):
                log.append(dict(issue=f"{c} frozen (flatline)", building=a, rows=len(g),
                                detail=f"{g.timestamp.min()} to {g.timestamp.max()} at {g[c].iloc[0]}: set to NaN"))
            df.loc[idx, c] = np.nan
    return df


def add_running_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes since fan start; the first 30 min are start-up transients (SAT falls from room
    temperature to setpoint) and carry no fault information."""
    df = df.sort_values(["ahu_id", "timestamp"]).reset_index(drop=True)
    run = df["fan_status"].eq(1)
    start = run & ~run.groupby(df["ahu_id"]).shift(fill_value=False)
    block = start.groupby(df["ahu_id"]).cumsum()
    first = df.groupby([df["ahu_id"], block])["timestamp"].transform("min")
    df["min_since_start"] = np.where(run, (df["timestamp"] - first).dt.total_seconds() / 60, np.nan)
    df["steady"] = run & (df["min_since_start"] >= 30)
    return df


def load_clean(return_log: bool = False):
    from .config import load_metadata
    meta = load_metadata()
    log: list = []
    df = load_all_raw()
    df = fix_clock_shift(df, log)
    df = drop_exact_duplicates(df, log)
    df = fix_power_units(df, meta, log)
    df = mask_out_of_range(df, log)
    df = mask_flatlines(df, log)
    df = add_running_flags(df)
    gaps = _gap_summary(df)
    log.extend(gaps)
    return (df, pd.DataFrame(log)) if return_log else df


def _gap_summary(df: pd.DataFrame) -> list:
    """Report (do not fill) gaps of 1 h or more that hit a whole building at once."""
    out = []
    for b, g in df.groupby(df["ahu_id"].str[:2]):
        t = pd.Series(sorted(g["timestamp"].unique()))
        step = t.diff()
        for i in step[step > pd.Timedelta("1h")].index:
            out.append(dict(issue="logging gap", building=b, rows=0,
                            detail=f"{t[i-1]} to {t[i]} ({step[i]}) missing for every AHU"))
    return out
