# Question 2: Modelling (summary of notebook sections 7 to 21)

**Decision.** Ship the residual model below, with its 12-hour alarm hold, to a pilot in listening mode. As shipped and tuned inside the folds it finds 27 of 28 labelled faults, names 21 correctly and raises 3 false alarms in 240 AHU-months, against 14 found and 8 false alarms for today's BMS alarm.

**Framing.** The model works in layers:

1. **Physics features** per row: setpoint error, outdoor-air fraction, coil ΔT, and fan power vs the affinity law.
2. **Fleet normal-behaviour models** (LightGBM, one per signal, AHU as a categorical input), trained on rows with no work order. They predict what each signal *should* be, and the residuals are the fault evidence. Residuals absorb unit differences, and the physics keeps alerts explainable.
3. **A LightGBM multiclass classifier** (normal + 4 faults) on 3 h and 12 h trailing medians of the residuals and physics features, plus time-share, correlation and trend features.
4. **A causal alarm state machine.** An alarm opens when P(fault), smoothed over 1 h, rises above 0.8, and closes after 12 h below 0.3. The type is the class with the highest mean probability since the alarm opened.

**Validation.** Leave-one-month-out on January to June with a 2-day purge, so neither the normal models nor the classifier see the scored month. Scoring is by event (found, right type, delay to first alarm, false alarms per AHU-month), and thresholds are tuned with `J = typed recall − 4 × FA/AHU-month`. These are known-unit numbers. Leakage across units is tested apart, by holding out a quarter of the AHUs at a time (the New AHU rows). Their 14-day window is the first stretch with no work order, so the +14-day row is optimistic.

**Results** (out of fold, 28 events, 240 AHU-months, 95% intervals). The bold row is the shipped setting (12-hour hold) with thresholds tuned inside each fold, the next row also lets the folds pick the hold, and every other row tunes on the scored months.

| | Found | Right type | Median delay | False alarms (per AHU-month) |
|---|---|---|---|---|
| **Model as shipped (nested)** | 27/28 (82-100%) | 21/28 (55-89%) | 10.0 h | 3 (0.003-0.037) |
| Model (nested, hold tuned per fold) | 27/28 | 20/28 (51-87%) | 11.3 h | 2 (0.008) |
| Thresholds tuned on scored months | 27/28 | 22/28 (59-92%) | 9.3 h | 2 (0.008) |
| Logistic regression | 25/28 (72-98%) | 17/28 (41-78%) | 6.8 h | 27 (0.074-0.164) |
| Expert rules (in-sample) | 19/28 (48-84%) | 10/28 (19-56%) | 2.6 h | 9 (0.017-0.071) |
| BMS alarm (today) | 14/28 | none | 2.9 h | 8 (0.033) |
| No CO2 (or airflow) | 28/28 (27) | 22/28 (21) | 10.6 h (11.2 h) | 1 (3) |
| No mixed-air sensor | 27/28 | 20/28 | 12.3 h | 8 (0.033) |
| New AHU, day 1 | 28/28 | 12/28 | 0.6 h | 179 (0.75) |
| New AHU, +14 days | 28/28 | 20/28 | 13.0 h | 8 (0.036) |

**Against the rules** the model finds 8 faults they miss and they find none it misses (McNemar p = 0.008; right type 12 to 0, p = 0.0005), with equal delay on the 19 both find. Both gaps and the one against the BMS alarm survive Holm. **Against a logistic regression** it cuts false alarms from 27 to 2, while its typing edge (6 to 1) is not significant (p = 0.125). **Across five seeds** every run finds 27/28, types 21 or 22 and raises 2 or 3 false alarms. **In money** (Yuill and Braun's figure of merit, PLN tariff, local labour rates) the nested model is worth Rp16,700 per AHU-month against −Rp2,100 for the BMS alarm, a lower bound since comfort is not priced.

**Errors.** Half of the in-sample misses and wrong types sit on two of the three chronically saturated units (B1-AHU-05 and B3-AHU-02), where a valve already at 100% makes three faults look alike. A saturated-unit mode and a site check should help, but are untested. The other confusions (B1-AHU-02, B3-AHU-06) need an independent discharge or zone temperature. The missed damper fault is in the mall, which has no CO2 sensor, and one of the false alarms looks like a real but unlabelled stuck damper (B1-AHU-11). **Behaviour tests.** Five of seven pass. One symptom alone (SAT 3 °C high, or no outdoor air) stays under P = 0.05, so the BMS alarm stays on.

**New AHU and fewer sensors.** Losing CO2 or airflow costs almost nothing, while losing the mixed-air temperature costs 6 more false alarms (it carries the damper evidence). A brand-new AHU is the real risk (179 false alarms on day one). Refitting the normal models on 14 days of history brings that to 8 (Q3 and Q4). **Test period.** `predictions.csv` (711,994 rows) passes the validator and holds 11 alarms longer than 6 h.
