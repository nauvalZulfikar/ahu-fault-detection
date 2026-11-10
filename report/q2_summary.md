# Question 2: Modelling (summary; full work in `code/notebooks/ahu_fault_detection.ipynb`, sections 7 to 21)

**Framing.** The model works in layers:

1. **Physics features** per row: setpoint error, outdoor-air fraction, coil ΔT, and fan power vs the affinity law.
2. **Fleet normal-behaviour models** (LightGBM, one per signal, AHU as a categorical input), trained on rows with no work order. They predict what each signal *should* be, and the residuals are the fault evidence.
3. **A LightGBM multiclass classifier** (normal + 4 faults) on 3 h and 12 h trailing medians of the residuals and physics features.
4. **A causal alarm state machine.** An alarm opens when P(fault), smoothed over 1 h, rises above 0.8. It closes after 12 h below 0.3. The type is the class with the highest mean probability since the alarm opened.

Why this design? Residuals absorb the unit differences that would force per-unit rule thresholds or teach a raw-signal classifier the unit instead of the fault, and the physics keeps alerts explainable.

**Validation.**
- **Leave-one-month-out** on January to June with a 2-day purge. Neither the normal models nor the classifier see the scored month.
- **No event leakage.** Events spanning two months leave training in both folds; rows within a day of an event never count as "normal".
- **Scored by event:** found, right type, delay to the first alarm, false alarms per AHU-month.
- **Thresholds** were tuned on the out-of-fold probabilities with `J = typed recall − 4 × FA/AHU-month`.

**Results** (out of fold, 28 events, 240 AHU-months, 95% intervals). The bold row picks alarm thresholds inside each fold; every other row picks them on the scored months.

| | Found | Right type | Median delay | False alarms (per AHU-month) |
|---|---|---|---|---|
| **Model (nested, honest)** | 27/28 (82-100%) | 20/28 (51-87%) | 11.3 h | 2 (0.001-0.030) |
| Model (in-sample) | 27/28 | 22/28 (59-92%) | 9.3 h | 2 (0.008) |
| Logistic regression | 25/28 (72-98%) | 17/28 (41-78%) | 6.8 h | 27 (0.074-0.164) |
| Expert rules (in-sample) | 19/28 (48-84%) | 10/28 (19-56%) | 2.6 h | 9 (0.017-0.071) |
| BMS alarm (today) | 14/28 | none | 2.9 h | 8 (0.033) |
| No CO2 (or airflow) | 28/28 (27) | 22/28 (21) | 10.6 h | 1 (2) |
| New AHU, day 1 | 28/28 | 12/28 | 0.6 h | 179 (0.75) |
| New AHU, +14 days | 28/28 | 20/28 | 13.0 h | 8 (0.036) |

**Against the rules**, the model finds 8 faults they miss and they find none it misses (McNemar p = 0.008; right type 12 to 0, p = 0.0005), with equal delay on the 19 both find (2.5 h vs 2.6 h). These gaps and the one against the BMS alarm survive a Holm correction. **Against a logistic regression** on the same features the model cuts false alarms from 27 to 2; its typing edge (6 to 1) is not significant (p = 0.125). **Across five seeds** every run finds 27/28, types 21 or 22 and raises 2 or 3 false alarms.

**Value in money.** Priced with Yuill and Braun's (2017) figure of merit at assumed rates (3 kW per fault, 0.13 USD/kWh, 20 USD/h), the nested model is worth 0.61 USD per AHU-month against −2.78 for the BMS alarm, and it picks the same thresholds as `J` (notebook §18).

**Errors.**
- **Half of the 6 misses and wrong types are on the three chronically saturated units** (B1-AHU-05, B3-AHU-02, B3-AHU-09), where valve at 100% makes three faults look alike (SAT just rises). Fix: a saturated-unit mode and a site check.
- **The other confusions** (B1-AHU-02, B3-AHU-06) need an independent discharge or zone temperature.
- **The missed damper fault** (B3-AHU-05) is in the mall, which has no CO2 sensor. **One of the two false alarms is a real but unlabelled stuck damper** (B1-AHU-11, Q1 §1.5).

**New AHU and fewer sensors.** Missing CO2 or airflow costs almost nothing (the temperature balance carries the damper evidence). **A brand-new AHU is the real risk** (179 false alarms on day one); **fourteen days of listening** with the normal models refit fixes most of it, which drives onboarding in Q3 and Q4.

**Test period.** `predictions.csv` (711,994 rows) passes the validator and holds 12 sustained alarms (over 6 h).
