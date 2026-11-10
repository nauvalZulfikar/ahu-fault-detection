"""Baselines fire on the rules they state, and the validation folds keep held-out data out."""

import numpy as np
import pandas as pd

from ahu_fdd import baseline as B
from ahu_fdd import pipeline as PL


def rows(n=60, **cols):
    d = pd.DataFrame({"ahu_id": "B1-AHU-01",
                      "timestamp": pd.date_range("2025-02-03", periods=n, freq="5min")})
    base = dict(sat_err_med3h=0.0, valve_coil_corr_12h=0.6, valve_hi_3h=0.0, oaf_med3h=0.15,
                oa_damper_cmd_pct=20.0, co2_ppm=600.0)
    for k, v in {**base, **cols}.items():
        d[k] = v
    return d


def test_bms_alarm_fires_only_outside_the_band():
    d = rows(4, sat_err_med3h=[0.0, 1.4, 1.6, -2.0])
    p = B.bms_alarm(d)
    assert (p["normal"] == [1, 1, 0, 0]).all()


def test_expert_rules_types():
    assert (B.expert_rules(rows())["normal"] == 1).all()
    damper = B.expert_rules(rows(oa_damper_cmd_pct=60.0, oaf_med3h=0.01))
    assert (damper["oa_damper_stuck"].iloc[12:] == 1).all()
    valve = B.expert_rules(rows(sat_err_med3h=3.0, valve_coil_corr_12h=0.0))
    assert (valve["chw_valve_stuck"] == 1).all()
    coil = B.expert_rules(rows(sat_err_med3h=1.2, valve_hi_3h=0.9))
    assert (coil["low_coil_capacity"] == 1).all()


def test_attach_labels_end_is_exclusive():
    df = rows(24)
    lab = pd.DataFrame({"ahu_id": ["B1-AHU-01"], "fault_type": ["sat_sensor_bias"],
                        "start": [df.timestamp[4]], "end": [df.timestamp[8]]})
    y = PL.attach_labels(df, lab)
    assert (y[4:8] == "sat_sensor_bias").all() and y[8] == "normal" and y[3] == "normal"


def test_fold_mask_purges_month_and_touching_events():
    t = pd.date_range("2025-01-01", "2025-06-30 23:55", freq="1h")
    df = pd.DataFrame({"ahu_id": np.r_[["B1-AHU-01"] * len(t), ["B1-AHU-02"] * len(t)],
                       "timestamp": np.r_[t, t], "split": "train"})
    lab = pd.DataFrame({"ahu_id": ["B1-AHU-02"], "fault_type": ["chw_valve_stuck"],
                        "start": [pd.Timestamp("2025-02-25")], "end": [pd.Timestamp("2025-03-10")]})
    ok = PL.fold_mask(df, lab, 3)
    ts = df.timestamp
    assert not ok[ts.dt.month.eq(3)].any()
    assert not ok[ts.between("2025-02-27", "2025-02-28 23:00")].any()  # 2-day purge before March
    assert not ok[ts.between("2025-04-01", "2025-04-02 23:00")].any()  # and after
    ev_rows = (df.ahu_id == "B1-AHU-02") & ts.between("2025-02-23", "2025-02-26")
    assert not ok[ev_rows].any()  # the event touching March is removed in February too
    assert ok[(df.ahu_id == "B1-AHU-01") & ts.between("2025-02-23", "2025-02-26")].all()
    assert ok[ts.dt.month.eq(1)].all()
