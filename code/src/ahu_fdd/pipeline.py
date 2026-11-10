"""End to end: clean -> physics -> normal-behaviour residuals -> windows -> classifier -> alarms.

Validation is leave-one-month-out on Jan-Jun with a purge: when month k is scored, nothing from
month k, nothing within 2 days of it, and no row of any fault event that touches it is used to
fit the classifier or the normal-behaviour model whose residuals score month k.

One indirect path remains. The classifier's training features for another month j are residuals
from a normal model fit without month j, and that model did see month k's healthy rows. So normal
behaviour from month k (never its labels) can reach the classifier. Production has no such path,
because the normal models there only ever see history.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F
from .clean import load_clean
from .config import CLASSES, FAULTS, OUT_DIR, load_labels, load_metadata

PURGE = pd.Timedelta("2D")
CLF_PARAMS = dict(objective="multiclass", num_class=len(CLASSES), n_estimators=300, learning_rate=0.05,
                  num_leaves=15, min_child_samples=200, subsample=0.5, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=8)
NORMAL_KEEP = 4  # keep 1 in 4 normal rows when fitting the classifier (speed; faults kept in full)
POST = dict(t_on=0.8, t_off=0.3, smooth=12, hold=144)  # tuned on out-of-fold data (notebook section 9)


def attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    out = pd.Series("normal", index=df.index, dtype=object)
    for e in labels.itertuples(index=False):
        m = (df["ahu_id"] == e.ahu_id) & (df["timestamp"] >= e.start) & (df["timestamp"] < e.end)
        out[m] = e.fault_type
    return out


def near_event(df: pd.DataFrame, labels: pd.DataFrame, margin=pd.Timedelta("1D")) -> pd.Series:
    """Rows within `margin` of a labelled event: start/end dates are engineers' estimates, so
    these rows are neither clean normal nor clean fault."""
    out = pd.Series(False, index=df.index)
    for e in labels.itertuples(index=False):
        out |= (df["ahu_id"] == e.ahu_id) & (df["timestamp"] >= e.start - margin) & (df["timestamp"] < e.end + margin)
    return out


def build_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, log = load_clean(return_log=True)
    meta = load_metadata()
    labels = load_labels()
    df = F.add_physics(df, meta)
    m = meta.set_index("ahu_id")
    df["has_co2"] = df["ahu_id"].map(m["has_co2_sensor"]).astype(int)
    df["has_flow"] = df["ahu_id"].map(m["has_airflow_station"]).astype(int)
    df["label"] = attach_labels(df, labels)
    df.loc[df["split"] == "test", "label"] = np.nan
    df["near_event"] = near_event(df, labels)
    df["month"] = df["timestamp"].dt.month
    return df, labels, log


def fold_mask(df: pd.DataFrame, labels: pd.DataFrame, month: int) -> pd.Series:
    """Training rows allowed when month `month` is held out."""
    lo = pd.Timestamp(2025, month, 1)
    hi = lo + pd.offsets.MonthBegin(1)
    ok = (df["split"] == "train") & ~df["timestamp"].between(lo - PURGE, hi + PURGE, inclusive="left")
    touching = labels[(labels.start < hi) & (labels.end > lo)]
    for e in touching.itertuples(index=False):
        ok &= ~((df["ahu_id"] == e.ahu_id) & (df["timestamp"] >= e.start - PURGE) & (df["timestamp"] < e.end + PURGE))
    return ok


def residuals(df: pd.DataFrame, labels: pd.DataFrame, months=range(1, 7)) -> pd.DataFrame:
    """Out-of-fold residuals for train months, residuals from a model fit on all of train for test."""
    steady = df["steady"]
    clean_normal = steady & (df["label"] == "normal") & ~df["near_event"]
    res = pd.DataFrame(index=df.index, columns=list(F.RESID.values()), dtype=float)
    for k in months:
        nm = F.NormalModels(df["ahu_id"].unique()).fit(df[clean_normal & fold_mask(df, labels, k)])
        rows = steady & (df["split"] == "train") & (df["month"] == k)
        res.loc[rows] = nm.residuals(df[rows]).values
    nm = F.NormalModels(df["ahu_id"].unique()).fit(df[clean_normal])
    rows = steady & (df["split"] == "test")
    res.loc[rows] = nm.residuals(df[rows]).values
    return res


def make_windows(df: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    s = df[df["steady"]].join(res).sort_values(["ahu_id", "timestamp"])
    return F.add_windows(s)


def training_sample(X: pd.DataFrame, y: pd.Series):
    """Subsampled rows, class codes and softened class weights shared by every classifier."""
    keep = (y != "normal") | (np.arange(len(y)) % NORMAL_KEEP == 0)
    X, y = X[keep], y[keep]
    counts = y.value_counts()
    w = y.map(lambda c: len(y) / (len(counts) * counts[c])) ** 0.5  # soften class imbalance
    return X, pd.Categorical(y, categories=CLASSES).codes, w


def fit_classifier(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> lgb.LGBMClassifier:
    X, codes, w = training_sample(X, y)
    clf = lgb.LGBMClassifier(**CLF_PARAMS, random_state=seed)
    clf.fit(X, codes, sample_weight=w)
    return clf


def predict_proba(clf, X) -> pd.DataFrame:
    return pd.DataFrame(clf.predict_proba(X), index=X.index, columns=CLASSES)


def oof_probabilities(w: pd.DataFrame, df: pd.DataFrame, labels: pd.DataFrame, cols: list[str],
                      months=range(1, 7), seed: int = 0, fit=fit_classifier) -> pd.DataFrame:
    tr = w[w["split"] == "train"]
    train_rows = tr[~tr["near_event"] | (tr["label"] != "normal")]  # drop fuzzy edges of events
    out = []
    for k in months:
        ok = fold_mask(df, labels, k).reindex(train_rows.index).fillna(False)
        clf = fit(train_rows.loc[ok, cols], train_rows.loc[ok, "label"], seed)
        hold = tr[tr["month"] == k]
        out.append(predict_proba(clf, hold[cols]))
    return pd.concat(out).reindex(tr.index)


def to_submission(df: pd.DataFrame, w: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Expand alarm labels from steady rows to every test row. Rows in the first 30 min after a
    start, and fan-off rows, carry the AHU's last state forward (unscored when the fan is off)."""
    t = df[df["split"] == "test"][["ahu_id", "timestamp"]].copy()
    t["fault_type"] = pred.reindex(t.index)
    t["fault_type"] = t.groupby("ahu_id")["fault_type"].ffill().fillna("normal")
    return t.sort_values(["ahu_id", "timestamp"]).reset_index(drop=True)


def run(out_path=None, post=POST):
    from .events import alarms
    df, labels, _ = build_frame()
    res = residuals(df, labels)
    w = make_windows(df, res)
    cols = F.feature_columns(w)
    tr = w[(w["split"] == "train") & (~w["near_event"] | (w["label"] != "normal"))]
    clf = fit_classifier(tr[cols], tr["label"])
    te = w[w["split"] == "test"]
    prob = predict_proba(clf, te[cols])
    pred = alarms(prob, te[["ahu_id", "timestamp"]], **post)
    sub = to_submission(df, w, pred)
    out_path = out_path or OUT_DIR.parent.parent / "predictions.csv"
    sub.to_csv(out_path, index=False)
    return sub, clf, cols
