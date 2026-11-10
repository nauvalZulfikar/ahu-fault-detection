"""Trailing windows must only look backwards, so the same code can run live."""

import numpy as np
import pandas as pd

from ahu_fdd import features as F


def window_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in F.ROLL_COLS})
    d["chw_valve_cmd_pct"] = rng.uniform(0, 100, n)
    d["ahu_id"] = np.where(np.arange(n) < n // 2, "B1-AHU-01", "B1-AHU-02")
    d["timestamp"] = pd.Timestamp("2025-01-06") + pd.to_timedelta(np.arange(n) % (n // 2) * 5, unit="min")
    return d


def test_windows_do_not_use_future_rows():
    d = window_frame()
    k = 150  # inside the first AHU
    before = F.add_windows(d.copy())
    future = d.copy()
    later = (future.ahu_id == "B1-AHU-01") & (future.index > k)
    future.loc[later, F.ROLL_COLS + ["chw_valve_cmd_pct"]] *= 100
    after = F.add_windows(future)
    new_cols = [c for c in before.columns if c not in d.columns]
    pd.testing.assert_frame_equal(before.loc[:k, new_cols], after.loc[:k, new_cols])


def test_windows_do_not_leak_across_ahus():
    d = window_frame()
    other = d.copy()
    other.loc[other.ahu_id == "B1-AHU-01", F.ROLL_COLS] += 50
    a, b = F.add_windows(d.copy()), F.add_windows(other)
    second = d.ahu_id == "B1-AHU-02"
    pd.testing.assert_series_equal(a.loc[second, "sat_err_med3h"], b.loc[second, "sat_err_med3h"])


def test_feature_columns_are_windows_and_flags_only():
    cols = F.feature_columns(F.add_windows(window_frame()).assign(building_code=1, has_co2=1, has_flow=0))
    assert cols[:3] == ["building_code", "has_co2", "has_flow"]
    assert all(c.endswith(("3h", "12h", "_trend")) for c in cols[3:])
    assert not set(cols) & set(F.ROLL_COLS)  # no raw, same-row signal goes to the classifier
