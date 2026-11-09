"""Features in three layers.

1. Physics per row: quantities a commissioning engineer would look at (setpoint error, mixing
   fraction, coil temperature drop, fan power vs the affinity law).
2. Residuals from models of normal behaviour: what this AHU usually does in these conditions,
   learned from rows with no open work order. A fault shows up as a persistent residual.
3. Trailing windows over the last 3 and 12 running hours, so the classifier sees persistence
   rather than single noisy readings. Windows only look backwards, so the same code runs live.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import physics as P

# ------------------------------------------------------------------ 1. physics per row


def add_physics(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.set_index("ahu_id")
    rated = df["ahu_id"].map(m["fan_rated_kw"])
    design = df["ahu_id"].map(m["design_airflow_m3h"])
    df["sat_err"] = df["sat_c"] - df["sat_sp_c"]
    df["coil_dt"] = df["mat_c"] - df["sat_c"]
    df["mix_dt"] = df["mat_c"] - df["rat_c"]
    df["oaf"] = np.clip(P.outdoor_air_fraction(df["mat_c"], df["rat_c"], df["oat_c"]), -0.5, 1.5)
    df["static_err"] = df["supply_static_pa"] - df["supply_static_sp_pa"]
    df["approach"] = df["sat_c"] - df["chwst_c"]
    df["power_ratio"] = df["fan_power_kw"] / P.fan_affinity_power(df["fan_speed_pct"].clip(lower=20), rated, 2.5)
    # airflow: measured in B2, otherwise proportional to fan speed (fan law, constant system curve)
    df["flow_est"] = df["sa_flow_m3h"].fillna(design * df["fan_speed_pct"] / 100)
    df["cool_kw"] = P.sensible_cooling_kw(df["flow_est"], df["mat_c"], df["sat_c"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["dow"] = df["timestamp"].dt.dayofweek
    df["building_code"] = df["ahu_id"].str[1].astype(int)
    return df


# ------------------------------------------------------------------ 2. normal-behaviour models
# target -> inputs. Inputs are things a fault does not change (weather, schedule, setpoints,
# central plant), plus the controller signals that are upstream of the target.
BASELINES = {
    "chw_valve_cmd_pct": ["mat_c", "oa_rh_pct", "fan_speed_pct", "chwst_c", "hour"],
    "fan_speed_pct": ["oat_c", "oa_rh_pct", "hour", "dow", "min_since_start"],
    "mat_c": ["oat_c", "rat_c", "oa_damper_cmd_pct"],
    "oa_damper_cmd_pct": ["hour", "dow", "min_since_start", "oat_c"],
    "co2_ppm": ["hour", "dow", "min_since_start"],
    "rat_c": ["oat_c", "oa_rh_pct", "hour", "dow", "min_since_start"],
}
RESID = {"chw_valve_cmd_pct": "r_valve", "fan_speed_pct": "r_fan", "mat_c": "r_mat",
         "oa_damper_cmd_pct": "r_damper", "co2_ppm": "r_co2", "rat_c": "r_rat"}
LGB_PARAMS = dict(objective="l1", n_estimators=150, learning_rate=0.1, num_leaves=31,
                  min_child_samples=50, subsample=0.5, subsample_freq=1, verbose=-1, n_jobs=8)


class NormalModels:
    """One gradient-boosted model per target for the whole fleet, with the AHU as a categorical
    input. Sharing one model lets similar units borrow strength; the AHU input lets each unit
    keep its own operating point (e.g. a unit whose valve always sits at 100%)."""

    def __init__(self, ahu_ids):
        self.cats = pd.CategoricalDtype(sorted(ahu_ids))
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def _x(self, df, cols):
        x = df[cols].copy()
        x["ahu"] = df["ahu_id"].astype(self.cats)
        return x

    def fit(self, df: pd.DataFrame) -> "NormalModels":
        for target, cols in BASELINES.items():
            d = df[df[target].notna()]
            if len(d) > 1000:
                self.models[target] = lgb.LGBMRegressor(**LGB_PARAMS).fit(self._x(d, cols), d[target])
        return self

    def residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=df.index, columns=list(RESID.values()))
        for target, model in self.models.items():
            out[RESID[target]] = df[target] - model.predict(self._x(df, BASELINES[target]))
        return out


# ------------------------------------------------------------------ 3. trailing windows
ROLL_COLS = ["sat_err", "r_valve", "r_fan", "r_mat", "r_damper", "r_co2", "r_rat", "oaf", "mix_dt",
             "static_err", "power_ratio", "coil_dt"]
WINDOWS = {"3h": 36, "12h": 144}  # in running 5-min rows


def add_windows(df: pd.DataFrame) -> pd.DataFrame:
    """df: steady running rows only, sorted by ahu_id, timestamp."""
    g = df.groupby("ahu_id", sort=False)
    new = {}
    for name, n in WINDOWS.items():
        for c in ROLL_COLS:
            new[f"{c}_med{name}"] = g[c].transform(lambda s: s.rolling(n, min_periods=n // 3).median())
        new[f"valve_hi_{name}"] = g["chw_valve_cmd_pct"].transform(lambda s: (s > 95).rolling(n, min_periods=n // 3).mean())
        new[f"valve_lo_{name}"] = g["chw_valve_cmd_pct"].transform(lambda s: (s < 5).rolling(n, min_periods=n // 3).mean())
        new[f"sat_off_{name}"] = g["sat_err"].transform(lambda s: (s.abs() > 1.5).rolling(n, min_periods=n // 3).mean())
    # does the coil respond to the valve? correlation of command and coil temperature drop
    new["valve_coil_corr_12h"] = g[["chw_valve_cmd_pct", "coil_dt"]].apply(
        lambda x: x["chw_valve_cmd_pct"].rolling(144, min_periods=48).corr(x["coil_dt"])).reset_index(level=0, drop=True)
    new["sat_err_std_3h"] = g["sat_err"].transform(lambda s: s.rolling(36, min_periods=12).std())
    new["r_valve_trend"] = new["r_valve_med3h"] - new["r_valve_med12h"]
    new["r_fan_trend"] = new["r_fan_med3h"] - new["r_fan_med12h"]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = ["building_code", "has_co2", "has_flow"]
    win = [c for c in df.columns if any(c.endswith(s) for s in ("3h", "12h", "_trend"))]
    return base + win
