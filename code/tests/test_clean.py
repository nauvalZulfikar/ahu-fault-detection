"""Cleaning rules on small synthetic frames: each rule fixes what it should, and never drops
or invents rows except the one rule whose job is to drop exact duplicates."""

import numpy as np
import pandas as pd

from ahu_fdd import clean as C


def frame(n=48, ahu="B1-AHU-01", start="2025-03-03 08:00", **cols):
    d = pd.DataFrame({"ahu_id": ahu, "timestamp": pd.date_range(start, periods=n, freq="5min"),
                      "split": "train", "fan_status": 1})
    for k, v in cols.items():
        d[k] = v
    return d


def test_exact_duplicates_dropped_and_logged():
    d = frame(10, sat_c=13.0)
    d = pd.concat([d, d.iloc[[2, 5]]]).reset_index(drop=True)
    log = []
    out = C.drop_exact_duplicates(d, log)
    assert len(out) == 10
    assert not out.duplicated(["ahu_id", "timestamp"]).any()
    assert log[0]["rows"] == 2


def test_power_in_watts_divided_for_the_whole_day_only():
    meta = pd.DataFrame({"ahu_id": ["B3-AHU-04"], "fan_rated_kw": [10.0]})
    d = frame(24 * 12 * 2, ahu="B3-AHU-04", start="2025-04-01 00:00", fan_power_kw=6.0)
    day1 = d.timestamp.dt.day == 1
    d.loc[day1, "fan_power_kw"] = 6000.0
    d.loc[day1 & (d.timestamp.dt.hour == 3), "fan_power_kw"] = 40.0  # standby power in W, under 5x rating
    out = C.fix_power_units(d.copy(), meta, [])
    assert len(out) == len(d)
    assert np.allclose(out.loc[day1 & (d.timestamp.dt.hour != 3), "fan_power_kw"], 6.0)
    assert np.allclose(out.loc[day1 & (d.timestamp.dt.hour == 3), "fan_power_kw"], 0.04)
    assert np.allclose(out.loc[~day1, "fan_power_kw"], 6.0)


def test_co2_limits_masked_not_dropped():
    d = frame(4, co2_ppm=[0.0, 600.0, 5000.0, 4999.0])
    out = C.mask_out_of_range(d.copy(), [])
    assert len(out) == 4
    assert out.co2_ppm.isna().tolist() == [True, False, True, False]


def test_frozen_sensor_masked_and_row_kept():
    sat = 13.0 + np.r_[np.random.default_rng(0).normal(0, 0.2, 20), np.zeros(40), np.random.default_rng(1).normal(0, 0.2, 20)]
    sat[20:60] = 13.1
    d = frame(80, sat_c=sat, mat_c=24 + np.arange(80) * 0.01, rat_c=23 + np.arange(80) * 0.01,
              co2_ppm=500 + np.arange(80), sa_flow_m3h=np.nan, filter_dp_pa=80 + np.arange(80))
    log = []
    out = C.mask_flatlines(d.copy(), log)
    assert len(out) == 80
    assert out.sat_c.iloc[20:60].isna().all()
    assert out.sat_c.drop(range(20, 60)).notna().all()
    assert out.mat_c.notna().all()  # other signals of the same rows untouched
    assert log and log[0]["rows"] == 40


def test_short_constant_run_is_not_a_flatline():
    d = frame(30, sat_c=13.0, mat_c=24.0, rat_c=23.0, co2_ppm=500.0, sa_flow_m3h=np.nan, filter_dp_pa=80.0)
    out = C.mask_flatlines(d.copy(), [])  # 30 rows < 36 (3 h)
    assert out.sat_c.notna().all()


def test_running_flags_mark_first_30_minutes_as_transient():
    d = frame(20)
    d.loc[:3, "fan_status"] = 0
    out = C.add_running_flags(d)
    assert out.loc[:3, "steady"].eq(False).all() and out.loc[:3, "min_since_start"].isna().all()
    assert out.loc[4:9, "steady"].eq(False).all()  # 0 to 25 min after start
    assert out.loc[10:, "steady"].all()  # 30 min and later
    assert out.loc[10, "min_since_start"] == 30


def test_clock_shift_restores_true_time():
    """Rows with true time in [A, B) were written one span early (UTC instead of local)."""
    span = pd.Timedelta("1h")
    true_t = pd.date_range("2025-06-01 00:00", periods=24 * 12, freq="5min")
    A, B = true_t[60], true_t[180]
    written = np.where((true_t >= A) & (true_t < B), true_t - span, true_t)
    d = pd.DataFrame({"ahu_id": "B1-AHU-01", "timestamp": written, "split": "train",
                      "true_t": true_t, "oat_c": np.arange(len(true_t), dtype=float)})
    log = []
    out = C.fix_clock_shift(d, log)
    assert len(out) == len(d)
    assert (out.timestamp.values == out.true_t.values).all()
    assert log[0]["rows"] == 120


def test_no_duplicates_means_no_clock_change():
    d = frame(50, oat_c=30.0)
    out = C.fix_clock_shift(d.copy(), [])
    assert out.timestamp.equals(d.timestamp)
