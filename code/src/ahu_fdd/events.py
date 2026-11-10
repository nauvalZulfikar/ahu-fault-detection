"""Turn row probabilities into alarms, and score alarms the way a maintenance team sees them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAULTS


# ------------------------------------------------------------------ alarms (causal)
def alarms(prob: pd.DataFrame, keys: pd.DataFrame, t_on: float, t_off: float, smooth: int,
           hold: int) -> pd.Series:
    """prob: class probabilities per steady running row (columns normal + FAULTS), keys: ahu_id,
    timestamp for the same rows, sorted by ahu_id, timestamp.

    Per AHU, in time order and using only the past:
      score  = rolling mean of P(fault) over the last `smooth` rows
      alarm opens when score > t_on, and closes after `hold` consecutive rows with score < t_off
      type   = fault class with the highest mean probability since the alarm opened
    Returns the predicted fault_type per row ("normal" outside alarms).
    """
    out = np.full(len(prob), "normal", dtype=object)
    pf = 1 - prob["normal"].to_numpy()
    pk = prob[FAULTS].to_numpy()
    ahu = keys["ahu_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ahu[1:] != ahu[:-1]])
    ends = np.r_[starts[1:], len(ahu)]
    for s, e in zip(starts, ends):
        score = pd.Series(pf[s:e]).rolling(smooth, min_periods=1).mean().to_numpy()
        on, below, acc = False, 0, None
        for i in range(e - s):
            if not on and score[i] > t_on:
                on, below, acc = True, 0, np.zeros(len(FAULTS))
            if on:
                acc += pk[s + i]
                out[s + i] = FAULTS[int(acc.argmax())]
                below = below + 1 if score[i] < t_off else 0
                if below >= hold:
                    on = False
    return pd.Series(out, index=prob.index)


# ------------------------------------------------------------------ event scoring
def runs(keys: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Contiguous runs of the same non-normal label per AHU (in running-row order)."""
    d = keys[["ahu_id", "timestamp"]].assign(label=label.values)
    new = (d["label"] != d["label"].shift()) | (d["ahu_id"] != d["ahu_id"].shift())
    d["run"] = new.cumsum()
    r = d[d["label"] != "normal"].groupby("run").agg(
        ahu_id=("ahu_id", "first"), fault_type=("label", "first"), start=("timestamp", "min"),
        end=("timestamp", "max"), rows=("label", "size"))
    return r.reset_index(drop=True)


def score_events(keys: pd.DataFrame, pred: pd.Series, labels: pd.DataFrame, ahu_months: float,
                 grace: str = "1D") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """keys/pred: running rows only. labels: event table restricted to the scored period.

    - detected: at least one alarm row inside [start, end)
    - delay: first alarm row inside the event minus its start
    - type correct: the most frequent alarm type inside the event is the true type
    - false alarm: an alarm run (any type, merged across type changes) that touches no labelled
      event, with `grace` after the repair so an alarm that closes a few hours late is not
      counted twice
    """
    d = keys[["ahu_id", "timestamp"]].assign(pred=pred.values)
    ev = []
    for e in labels.itertuples(index=False):
        x = d[(d.ahu_id == e.ahu_id) & (d.timestamp >= e.start) & (d.timestamp < e.end)]
        hit = x[x.pred != "normal"]
        ev.append(dict(ahu_id=e.ahu_id, fault_type=e.fault_type, start=e.start, end=e.end,
                       running_rows=len(x), detected=len(hit) > 0,
                       delay_h=(hit.timestamp.min() - e.start).total_seconds() / 3600 if len(hit) else np.nan,
                       pred_type=hit.pred.mode().iloc[0] if len(hit) else None))
    ev = pd.DataFrame(ev)
    ev = ev[ev.running_rows > 0]
    ev["type_ok"] = ev.detected & (ev.pred_type == ev.fault_type)

    any_alarm = pd.Series(np.where(pred.values != "normal", "fault", "normal"), index=keys.index)
    al = runs(keys, any_alarm)
    lab = labels.assign(end_g=labels.end + pd.Timedelta(grace))
    def touches(a):
        m = lab[(lab.ahu_id == a.ahu_id) & (lab.start <= a.end) & (lab.end_g >= a.start)]
        return len(m) > 0
    al["true_alarm"] = al.apply(touches, axis=1) if len(al) else []
    fa = int((~al["true_alarm"]).sum()) if len(al) else 0
    n = len(ev)
    summary = dict(events=n, detected=int(ev.detected.sum()), type_ok=int(ev.type_ok.sum()),
                   recall=ev.detected.mean(), typed_recall=ev.type_ok.mean(),
                   median_delay_h=ev.delay_h.median(), alarms=len(al), false_alarms=fa,
                   fa_per_ahu_month=fa / ahu_months)
    return summary, ev, al


def objective(s: dict, fa_weight: float = 4.0) -> float:
    """What we tune post-processing on: share of faults found with the right type, minus a
    penalty for false alarms per AHU-month (weight 4: 0.05 FA/AHU-month, i.e. about 2 wasted
    visits a month across 40 AHUs, costs as much as missing one fault in five)."""
    return s["typed_recall"] - fa_weight * s["fa_per_ahu_month"]


# ------------------------------------------------------------------ value in money
# Indonesian prices, Q4 2026. Every number has a source in the notebook (section 18).
# - tariff, Rp/kWh. PLN B-3 (business, medium voltage, above 200 kVA) Rp1.114,74 and B-2
#   (6.600 VA to 200 kVA) Rp1.444,70, unchanged for Oct-Dec 2026 (ESDM).
# - labour, Rp/hour, loaded. Monthly wage / 173 h x (1 + employer BPJS + THR). Employer BPJS is
#   JHT 3.7% + JP 2% + JKK 0.24-1.74% + JKM 0.3% + Kesehatan 4% (10.2-11.7%), THR is one month a
#   year (8.3%). Wages are UMK Denpasar 2026 Rp3.499.878, a typical building technician posting
#   Rp5 jt, and a manufacturer service technician Rp9 jt.
# - kw_tr, chiller plant kW per ton of cooling. 0.56 design, 0.62 measured in a West Java mall,
#   1.61 an old air-cooled plant (Indonesian audits).
# - response_h. 4 h is the critical tier of HVAC service contracts, 24 h the routine next-day
#   tier, 48 h the "2x24 jam" repair promise of a manufacturer's service in Indonesia.
# SERVICE_H stays an engineering estimate: contractor guides give 1-2 h for an actuator swap and
#   2-4 h for a motorised damper; no AHU-specific labour standard was found for Indonesia.
SERVICE_H = dict(chw_valve_stuck=2.0, oa_damper_stuck=3.0, sat_sensor_bias=1.5, low_coil_capacity=4.0)


def _loaded(wage: float, jkk: float) -> float:
    return round(wage / 173 * (1 + 0.037 + 0.02 + jkk + 0.003 + 0.04 + 1 / 12))


SCENARIOS = {"pessimistic": dict(kw_tr=0.56, tariff=1114.74, labour=_loaded(9_000_000, 0.0174), response_h=48.0),
             "central": dict(kw_tr=0.62, tariff=1114.74, labour=_loaded(5_000_000, 0.0089), response_h=24.0),
             "optimistic": dict(kw_tr=1.61, tariff=1444.70, labour=_loaded(3_499_878, 0.0024), response_h=4.0)}


def excess_kw(w: pd.DataFrame, labels: pd.DataFrame, pad: str = "1D", window: str = "28D") -> pd.DataFrame:
    """Extra coil cooling (kW thermal, sensible only) and extra fan power (kW electric) during each
    labelled event, against the same AHU's own healthy rows in the 28 days around it, matched on
    hour of day and weekday/weekend. Latent load is ignored, so the cooling side is conservative.
    w: steady rows with cool_kw and fan_power_kw."""
    w = w.assign(hour=w.timestamp.dt.hour, wkend=w.timestamp.dt.day_of_week >= 5)
    out = []
    for e in labels.itertuples(index=False):
        a = w[w.ahu_id == e.ahu_id]
        bad = pd.Series(False, index=a.index)
        for o in labels[labels.ahu_id == e.ahu_id].itertuples(index=False):
            bad |= (a.timestamp >= o.start - pd.Timedelta(pad)) & (a.timestamp < o.end + pd.Timedelta(pad))
        near = (a.timestamp >= e.start - pd.Timedelta(window)) & (a.timestamp < e.end + pd.Timedelta(window))
        base = a[~bad & near].groupby(["hour", "wkend"])[["cool_kw", "fan_power_kw"]].mean()
        base.columns = ["cool_b", "fan_b"]
        m = a[(a.timestamp >= e.start) & (a.timestamp < e.end)].join(base, on=["hour", "wkend"])
        m = m.dropna(subset=["cool_b"])
        out.append(dict(ahu_id=e.ahu_id, start=e.start, coil_kw=(m.cool_kw - m.cool_b).mean(),
                        fan_kw=(m.fan_power_kw - m.fan_b).mean()))
    return pd.DataFrame(out)


def value(ev: pd.DataFrame, al: pd.DataFrame, ahu_months: float, kw: pd.DataFrame, kw_tr: float,
          tariff: float, labour: float, response_h: float = 24.0, typed: bool = True,
          p_tech: float = 0.5) -> dict:
    """Net value in Rp per AHU-month, adapted from Yuill's figure of merit (Yuill 2014, ch. 7;
    Yuill & Braun 2017, eq. 7.1-7.21). The baseline is what the labels record: every fault is
    repaired at its labelled end anyway, so the repair itself is not counted, only what changes.

    - electric kW wasted per event = coil_kw * kw_tr / 3.517 + fan_kw from `excess_kw`, floored
      at 0 (a fault that cools less saves energy; its cost is comfort, which is not priced here)
    - right type: energy saved by repairing earlier, kW * tariff * running hours between
      (first alarm + response_h) and the labelled repair
    - wrong type: one wasted check of the predicted fault, SERVICE_H[pred] * labour, no benefit
    - missed: 0
    - false alarm: one wasted check at the mean service time
    - typed=False (BMS alarm, no diagnosis, Yuill eq. 7.21): the technician finds the fault with
      probability p_tech after 0.5 h, otherwise spends 2 h more for nothing
    """
    k = ev[["ahu_id", "start"]].merge(kw, on=["ahu_id", "start"], how="left")
    elec = (k.coil_kw * kw_tr / 3.517 + k.fan_kw).clip(lower=0).fillna(0).to_numpy()
    dur = (ev.end - ev.start).dt.total_seconds() / 3600
    saved_h = ((dur - ev.delay_h - response_h).clip(lower=0) * ev.running_rows / 12 / dur).fillna(0)
    energy = elec * tariff * saved_h.to_numpy()
    if typed:
        gain = np.where(ev.type_ok, energy, 0.0)
        visits = np.where(ev.detected & ~ev.type_ok, ev.pred_type.map(SERVICE_H).fillna(2.0) * labour, 0.0)
        fa_cost = np.mean(list(SERVICE_H.values())) * labour
    else:
        gain = np.where(ev.detected, p_tech * energy, 0.0)
        visits = np.where(ev.detected, 0.5 * labour + (1 - p_tech) * 2 * labour, 0.0)
        fa_cost = 2.5 * labour
    n_fa = int((~al.true_alarm).sum()) if len(al) else 0
    cost = visits.sum() + n_fa * fa_cost
    return dict(energy_rp=float(gain.sum()), visit_rp=float(cost),
                rp_per_ahu_month=float((gain.sum() - cost) / ahu_months))


# ------------------------------------------------------------------ uncertainty
def proportion_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for k successes out of n."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def rate_ci(k: int, exposure: float, level: float = 0.95) -> tuple[float, float]:
    """Exact Poisson interval for k counts over `exposure` (e.g. false alarms per AHU-month)."""
    from scipy.stats import chi2
    a = (1 - level) / 2
    lo = chi2.ppf(a, 2 * k) / 2 if k > 0 else 0.0
    return float(lo / exposure), float(chi2.ppf(1 - a, 2 * k + 2) / 2 / exposure)


def median_ci(x, n_boot: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap interval for the median, resampling events."""
    x = np.asarray(pd.Series(x).dropna())
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def mcnemar_exact(a_ok: pd.Series, b_ok: pd.Series) -> dict:
    """Paired comparison of two methods on the same events. Only events where they disagree count."""
    from scipy.stats import binomtest
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(only_first=b, only_second=c, p_value=float(p))
