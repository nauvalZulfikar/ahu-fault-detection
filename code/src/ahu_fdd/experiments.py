"""The two "what if" questions in Q2, run on cached out-of-fold data.

- new AHU: the AHU has no history. Its normal-behaviour models come from the rest of the fleet
  only, and the classifier never saw it.
- fewer sensors: the building has no CO2 sensor or airflow station.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import pipeline as PL
from .events import alarms, score_events


def score(prob, keys, labels, ahu_months, post=PL.POST):
    s, ev, _ = score_events(keys, alarms(prob, keys, **post), labels, ahu_months)
    return s, ev


def commissioning_window(df, labels, ahu, days=14):
    """First `days`-long stretch of an AHU's train data with no labelled event within a day of it:
    stands in for a two-week listening period on a unit an engineer has checked is healthy."""
    t0 = df.loc[df["ahu_id"] == ahu, "timestamp"].min()
    ev = labels[labels.ahu_id == ahu]
    while True:
        t1 = t0 + pd.Timedelta(days=days)
        clash = ev[(ev.start < t1 + pd.Timedelta("1D")) & (ev.end > t0 - pd.Timedelta("1D"))]
        if clash.empty:
            return t0, t1
        t0 = clash.end.max() + pd.Timedelta("1D")


def new_ahu(df, w, labels, n_groups=4, seed=0, baseline_days=0):
    """Hold out a quarter of the AHUs at a time (stratified by building). Normal models and the
    classifier are fit on the other AHUs only; the held-out AHUs are scored on all of Jan-Jun.

    baseline_days > 0: each held-out AHU first gets a listening period of that many days, and the
    normal models are refit with those days included (the unit's own operating point). The
    classifier still never sees the unit, and the listening period itself is not scored."""
    rng = np.random.default_rng(seed)
    ahus = sorted(df["ahu_id"].unique())
    group = {}
    for b in "123":
        ids = [a for a in ahus if a[1] == b]
        for i, a in enumerate(rng.permutation(ids)):
            group[a] = i % n_groups
    train = df["split"] == "train"
    clean_normal = df["steady"] & train & (df["label"] == "normal") & ~df["near_event"]
    cols = F.feature_columns(w)
    probs = []
    for k in range(n_groups):
        out = df["ahu_id"].map(group).ne(k)
        listen = pd.Series(False, index=df.index)
        if baseline_days:
            for a in df.loc[~out, "ahu_id"].unique():
                t0, t1 = commissioning_window(df, labels, a, baseline_days)
                listen |= (df["ahu_id"] == a) & df["timestamp"].between(t0, t1, inclusive="left")
        nm = F.NormalModels(ahus).fit(df[clean_normal & (out | listen)])
        rows = df["steady"] & train & ~out
        res = nm.residuals(df[rows])
        wk = F.add_windows(df[rows].join(res).sort_values(["ahu_id", "timestamp"]))
        wk = wk[~listen.reindex(wk.index)]
        tr = w[(w["split"] == "train") & w["ahu_id"].map(group).ne(k)]
        tr = tr[~tr["near_event"] | (tr["label"] != "normal")]
        clf = PL.fit_classifier(tr[cols], tr["label"])
        probs.append(PL.predict_proba(clf, wk[cols]).join(wk[["ahu_id", "timestamp"]]))
    p = pd.concat(probs).sort_values(["ahu_id", "timestamp"])
    return p.drop(columns=["ahu_id", "timestamp"]), p[["ahu_id", "timestamp"]]


def fewer_sensors(w, labels, drop=("co2", "flow")):
    """Re-run the out-of-fold classifier as if no AHU had CO2 and/or airflow measurement."""
    w = w.copy()
    if "co2" in drop:
        w[[c for c in w.columns if c.startswith("r_co2")]] = np.nan
        w["has_co2"] = 0
    if "flow" in drop:
        w["has_flow"] = 0  # flow only enters through flow_est, which falls back to fan speed
    cols = F.feature_columns(w)
    tr = w[w["split"] == "train"]
    return PL.oof_probabilities(w, w, labels, cols).loc[tr.index], tr[["ahu_id", "timestamp"]]
