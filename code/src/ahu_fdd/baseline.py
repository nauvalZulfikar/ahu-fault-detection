"""Two baselines without machine learning, scored with the same alarm logic and event scorer as the
model, so the comparison in the notebook is like for like. `fit_logistic` adds the simplest ML
baseline, a multinomial logistic regression on the same features, rows and weights as the model.

- bms_alarm: what most buildings run today. A supply-air temperature alarm when SAT is more than
  `sat_band` away from setpoint (3-hour median). It says "something is wrong", never which fault.
- expert_rules: fleet-wide engineering rules in the spirit of APAR (House et al., 2001), one rule
  per fault an engineer can write down. Thresholds were picked on Jan-Jun itself, which flatters
  the rules (the model is scored out of fold). Sensor bias has no rule: by design it leaves SAT on
  setpoint, so there is nothing for a fixed threshold to catch.
"""

from __future__ import annotations

import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLASSES
from .pipeline import training_sample


def _as_prob(label: pd.Series) -> pd.DataFrame:
    """One-hot "probabilities" so the rules go through the same alarm state machine."""
    return pd.DataFrame({c: (label == c).astype(float) for c in CLASSES}, index=label.index)


def bms_alarm(w: pd.DataFrame, sat_band: float = 1.5) -> pd.DataFrame:
    fire = w["sat_err_med3h"].abs() > sat_band
    return _as_prob(pd.Series("chw_valve_stuck", index=w.index).where(fire, "normal"))


def expert_rules(w: pd.DataFrame, sat_hi: float = 1.5, sat_warm: float = 1.0, corr_lo: float = 0.1,
                 oaf_lo: float = 0.05, damper_hi: float = 40, co2_hi: float = 1100) -> pd.DataFrame:
    """w: steady rows sorted by ahu_id, timestamp, with the trailing-window columns."""
    g = w.groupby("ahu_id", sort=False)
    med = lambda c: g[c].transform(lambda s: s.rolling(36, min_periods=12).median())
    damper_cmd, co2 = med("oa_damper_cmd_pct"), med("co2_ppm")
    sat_err, corr = w["sat_err_med3h"], w["valve_coil_corr_12h"]
    label = pd.Series("normal", index=w.index, dtype=object)
    # damper commanded open, yet no outdoor air arrives (temperature balance) or CO2 builds up
    label[(damper_cmd > damper_hi) & ((w["oaf_med3h"] < oaf_lo) | (co2 > co2_hi))] = "oa_damper_stuck"
    # valve saturated and SAT still warm, while the coil does respond to the valve: not enough coil
    label[(sat_err > sat_warm) & (w["valve_hi_3h"] > 0.8) & ~(corr < corr_lo)] = "low_coil_capacity"
    # SAT off setpoint and the coil temperature drop no longer follows the valve command
    label[(sat_err.abs() > sat_hi) & (corr < corr_lo)] = "chw_valve_stuck"
    return _as_prob(label)


def fit_logistic(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Multinomial logistic regression. Imputer, scaler and encoder sit inside the pipeline, so they
    learn from the training fold only. Missing values get an indicator column, because a missing
    sensor is information, not a zero."""
    X, codes, w = training_sample(X, y)
    num = [c for c in X.columns if c != "building_code"]
    prep = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["building_code"]),
        (make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                       StandardScaler()), num))
    clf = make_pipeline(prep, LogisticRegression(max_iter=500, random_state=seed))
    clf.fit(X, codes, logisticregression__sample_weight=w)
    return clf
