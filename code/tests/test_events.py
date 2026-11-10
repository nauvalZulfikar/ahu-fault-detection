"""Alarm state machine and event scorer, on hand-made probabilities and labels."""

import numpy as np
import pandas as pd
import pytest

from ahu_fdd import events as E
from ahu_fdd.config import CLASSES


def keys(n, ahu="B1-AHU-01", start="2025-01-06 00:00"):
    return pd.DataFrame({"ahu_id": ahu, "timestamp": pd.date_range(start, periods=n, freq="5min")})


def prob(p_fault, fault="chw_valve_stuck"):
    p = pd.DataFrame(0.0, index=range(len(p_fault)), columns=CLASSES)
    p["normal"] = 1 - np.asarray(p_fault)
    p[fault] = p_fault
    return p


def test_alarm_opens_on_threshold_and_holds():
    pf = np.r_[np.zeros(10), np.ones(20), np.zeros(30)]
    out = E.alarms(prob(pf), keys(60), t_on=0.8, t_off=0.3, smooth=1, hold=5)
    assert (out[:10] == "normal").all()
    assert (out[10:30] == "chw_valve_stuck").all()
    assert (out[30:35] == "chw_valve_stuck").all()  # stays open for `hold` quiet rows
    assert (out[35:] == "normal").all()


def test_alarm_is_causal():
    rng = np.random.default_rng(0)
    pf = rng.uniform(0, 1, 300)
    a = E.alarms(prob(pf), keys(300), 0.7, 0.3, 12, 36)
    pf2 = pf.copy()
    pf2[200:] = 1 - pf2[200:]
    b = E.alarms(prob(pf2), keys(300), 0.7, 0.3, 12, 36)
    assert (a[:200] == b[:200]).all()


def test_alarm_type_does_not_flicker():
    p = prob(np.ones(20))
    p.loc[:, "chw_valve_stuck"] = 0.6
    p.loc[:, "sat_sensor_bias"] = 0.4
    p.loc[15, ["chw_valve_stuck", "sat_sensor_bias"]] = [0.1, 0.9]  # one noisy row
    out = E.alarms(p, keys(20), 0.8, 0.3, 1, 5)
    assert (out == "chw_valve_stuck").all()


def scored_case():
    k = keys(3 * 288)  # three days
    labels = pd.DataFrame({"ahu_id": ["B1-AHU-01"], "fault_type": ["oa_damper_stuck"],
                           "start": [pd.Timestamp("2025-01-06 10:00")], "end": [pd.Timestamp("2025-01-06 20:00")]})
    pred = pd.Series("normal", index=k.index, dtype=object)
    pred[k.timestamp.between("2025-01-06 12:00", "2025-01-06 22:00")] = "oa_damper_stuck"  # late close, inside grace
    pred[k.timestamp.between("2025-01-08 06:00", "2025-01-08 07:00")] = "sat_sensor_bias"  # nothing there
    return k, pred, labels


def test_score_events_counts_detection_delay_type_and_false_alarms():
    k, pred, labels = scored_case()
    s, ev, al = E.score_events(k, pred, labels, ahu_months=1)
    assert s["detected"] == 1 and s["type_ok"] == 1
    assert ev.delay_h.iloc[0] == pytest.approx(2.0)
    assert s["alarms"] == 2 and s["false_alarms"] == 1
    assert s["fa_per_ahu_month"] == 1


def test_wrong_type_is_detected_but_not_typed():
    k, pred, labels = scored_case()
    pred[pred == "oa_damper_stuck"] = "chw_valve_stuck"
    s, _, _ = E.score_events(k, pred, labels, ahu_months=1)
    assert s["detected"] == 1 and s["type_ok"] == 0


def test_objective_penalises_false_alarms():
    assert E.objective(dict(typed_recall=0.8, fa_per_ahu_month=0.05)) == pytest.approx(0.6)


def test_intervals_and_mcnemar_match_reported_numbers():
    lo, hi = E.proportion_ci(27, 28)
    assert hi > 0.99 and 0.8 < lo < 0.83
    lo, hi = E.rate_ci(2, 240)
    assert lo == pytest.approx(0.001, abs=5e-4) and hi == pytest.approx(0.030, abs=1e-3)
    a = pd.Series([True] * 27 + [False])
    b = pd.Series([True] * 19 + [False] * 9)
    assert E.mcnemar_exact(a, b)["p_value"] == pytest.approx(0.0078, abs=1e-4)
