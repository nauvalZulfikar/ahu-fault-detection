# Question 2: Modelling (summary; full work in `code/notebooks/02_modelling.ipynb`)

**Framing.** The model works in layers:

1. **Physics features** per row: setpoint error, outdoor-air fraction, coil ΔT, and fan power vs the affinity law.
2. **Fleet normal-behaviour models** (LightGBM, one per signal, AHU as a categorical input), trained on rows with no work order. They predict what each signal *should* be, and the residuals are the fault evidence.
3. **A LightGBM multiclass classifier** (normal + 4 faults) on 3 h and 12 h trailing medians of the residuals and physics features.
4. **A causal alarm state machine.** An alarm opens when P(fault), smoothed over 1 h, rises above 0.8. It closes after 12 h below 0.3. The type is the class with the highest mean probability since the alarm opened.

Why this design? Pure rules need thresholds per unit, and a classifier on raw signals learns the unit, not the fault. Residuals absorb unit differences, and the physics keeps alerts explainable.

**Validation.**
- **Leave-one-month-out** on January to June with a 2-day purge. Neither the normal models nor the classifier see the scored month.
- **No event leakage.** An event that spans two months is removed from training in both folds, and rows within a day of an event are not used as "normal".
- **Scored by event, as the maintenance team sees it:** found or not, right type or not, delay to the first alarm, and false alarms per AHU-month.
- **Thresholds** were tuned on the out-of-fold probabilities with `J = typed recall − 4 × FA/AHU-month`.

**Results (out of fold, 28 labelled events, 240 AHU-months, 95% intervals in brackets)**

| | Found | Right type | Median delay | False alarms (per AHU-mo.) |
|---|---|---|---|---|
| **Model** | 27/28 (82-100%) | 22/28 (59-92%) | 9.3 h | 2 (0.001-0.030) |
| Expert rules (tuned in-sample) | 19/28 (48-84%) | 10/28 (19-56%) | 2.6 h | 9 (0.017-0.071) |
| BMS supply-air alarm (today) | 14/28 | none | 2.9 h | 8 (0.033) |
| No CO2 (or airflow) | 28/28 (27) | 22/28 (21) | 10.6 h | 1 (2) |
| New AHU, day 1 | 28/28 | 12/28 | 0.6 h | 179 (0.75) |
| New AHU, +14 days | 28/28 | 20/28 | 13.0 h | 8 (0.036) |

**Against the rules**, the model finds 8 faults they miss and they find none it misses (McNemar p = 0.008; right type 12 to 0, p = 0.0005). On the 19 faults both find, delay is equal (2.5 h vs 2.6 h). **Across five seeds** every run finds 27/28, types 21 or 22 and raises 2 or 3 false alarms.

**Errors.**
- **Half of the 6 misses and wrong types are on the three chronically saturated units** (B1-AHU-05, B3-AHU-02, B3-AHU-09). Their valve is already at 100%, so low coil capacity, a stuck-closed valve and sensor bias all show up the same way: SAT rises. Fix: a separate mode for saturated units, using SAT against CHW supply temperature and load, plus a site check of these units.
- **The other confusions** are sensor bias vs low coil capacity (B1-AHU-02) and stuck valve vs sensor bias (B3-AHU-06). Telling these apart needs an independent temperature (a discharge thermometer or zone temperatures).
- **The missed damper fault** (B3-AHU-05) is in the mall, which has no CO2 sensor. **One of the two false alarms is a real but unlabelled stuck damper** (B1-AHU-11, Q1 §1.5).

**New AHU and fewer sensors.**
- **Missing CO2 or airflow costs almost nothing.** The outdoor-air fraction from the temperature balance carries the damper evidence, so fewer sensors means the same model with those inputs missing.
- **A brand-new AHU is the real risk.** On day one the fleet models do not know the unit's operating point, which gives about 90 times the false-alarm rate of known units (179 false alarms instead of 2).
- **Fourteen days of listening fixes most of this** (data collected, no alerts sent, normal models refit with the new unit's healthy weeks). That rule drives onboarding in Q3 and the advice in Q4.

**Test period.** `predictions.csv` (711,994 rows) passes the starter validator. It holds 12 sustained alarms (longer than 6 h), for example a stuck valve on B1-AHU-01 from 7 to 15 July. B2-AHU-08's frozen SAT sensor is masked, not alarmed.
