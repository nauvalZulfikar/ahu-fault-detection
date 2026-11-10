# Question 3: Production architecture

![Architecture](figures/architecture.png){style="max-height:none"}

## 3.1 Data path

**Edge (one small agent per building).** A Windows service reads new rows from the BMS SQL Server every 5 minutes. It uses the timestamp plus the row key as a watermark, so each read is incremental. It maps the site's point names to the standard schema I define using a per-building mapping file (a YAML file kept in Git), writes compressed Parquet into a local buffer, and pushes the files over outbound HTTPS to an ingest API. The agent does no modelling. Because the PC only allows outbound traffic, the agent also polls the API for config and version updates, so I never need inbound access. The buffer holds 14 days, so an internet outage loses nothing: the agent re-sends when it reconnects and the ingest API ignores duplicates by (ahu_id, timestamp, hash). The agent sends a heartbeat every 5 minutes.

**Cloud storage.** Files land in object storage in two zones:
- *raw*: append-only and never edited, so any result can be replayed.
- *curated*: the output of the same `clean.py` used in this take-home, run as data-quality (DQ) checks. These cover clock shifts, duplicate exports, unit errors, out-of-range values, frozen sensors and gaps. Each check writes a DQ flag per AHU per hour instead of silently fixing data. Flags feed both the model, which gives no fault alert while a key sensor is flagged, and a DQ ticket queue, because a frozen sensor is itself worth a work order.

**Scale.** 600 AHUs at 12 rows an hour is about 173k rows a day, or about 2,000 AHUs and 576k rows a day at the stated target. That is a few hundred MB a day in Parquet. It fits comfortably in one Postgres/TimescaleDB instance or DuckDB over Parquet. I do not need Spark or streaming: an hourly batch job on one machine handles the whole fleet in minutes.

## 3.2 Models

**Training (weekly, or on demand).** The job retrains the fleet normal-behaviour models and the fault classifier on all curated history, with no-fault rows defined as rows without an open work order. It runs the same leave-one-month-out evaluation as the notebook. The output is one versioned bundle: model files, alarm thresholds, feature list, code commit and evaluation report, stored in a model registry (MLflow or a versioned bucket).

**Promotion.** False alarms are rare (0.0125 per AHU-month in Q2), so a one-week shadow run on 40 AHUs covers about 9 AHU-months and expects fewer than one false alarm from either bundle. A week cannot tell two bundles apart, so promotion has two gates. First, a *labelled replay*: the new bundle (challenger) and the live one (champion) score all labelled history with the same out-of-fold procedure, and the challenger must find and type at least as many events, compared event by event with McNemar's test, with no more false alarms. Second, a fleet-wide *shadow run* long enough to bound the false-alarm rate. With no unconfirmed alarm in 30 AHU-months, the 95% upper bound on the rate is 0.1 per AHU-month, which takes a little over three weeks on 40 AHUs and under two days on 600. Rollback means pointing back to the previous bundle ID, which takes minutes because bundles are never edited. The Q2 notebook writes such a bundle (`outputs/model/bundle.json` holds data hashes, library versions, seed, thresholds and code commit).

**Inference (hourly).** For each AHU the job takes the last 24 h of curated data, computes features, scores, and runs the alarm state machine. Its state (open alarms, cumulative type probabilities) persists between runs, so the logic is exactly the causal code validated offline. Hourly is enough because these faults develop over hours and cost money over days: the validated median delay is about 10 hours, set by the 3-12 h evidence windows, not by how often I score. Scoring every 5 minutes would add cost and alert churn and find nothing sooner. Safety-critical conditions (a fan trip, a fire damper) stay with the BMS's own alarms.

**Technology choices.** Python, LightGBM and Parquet, because that is what the model is built with and one engineer can run it. Postgres/TimescaleDB for curated data and alarm state, because the volume is small and SQL suits the facility engineers. A scheduler (Prefect, or cron on a managed container) runs the jobs. A managed cloud (Azure fits sites that already run Windows/SQL Server) means no servers to patch. Nothing in this design needs Kafka, Spark or a GPU.

## 3.3 From score to work order

Raw alarms do not go straight to the CMMS. They pass through a small rules layer:
1. **Deduplicate.** One open ticket per AHU and fault. A repeat alarm updates the existing ticket instead of opening a new one.
2. **Group by building.** If more than about 30% of a building's AHUs show the same signature at once (for example SAT high with valves at 100%), it is probably a plant problem such as warm chilled water. The layer raises one plant ticket and suppresses the AHU tickets.
3. **Severity and route.** Faults that waste energy or affect comfort now (stuck valve, stuck damper on a CO2-controlled unit) become work orders. Slow faults (coil capacity, sensor bias) go into a weekly digest. Each ticket carries the fault type, the evidence plots, and a short "what to check" note per fault type (e.g. *stroke the valve 0 to 100% and watch SAT*).
4. **CMMS REST API.** Tickets are created and updated through the client's CMMS. When a technician closes a ticket, the result (*confirmed / no fault found / other cause*) is read back. **These outcomes become new labels** and drive the precision metric. Without them the system cannot get better.

## 3.4 Monitoring

| What | How | Alert when |
|---|---|---|
| Data freshness | heartbeat and last row per building | no data for 30 min |
| Data quality | DQ flags per AHU per day | new flag type, or > 5% of rows flagged |
| Input drift | PSI of the 10 most important features and of weather and chilled-water temperature, live month vs training window | PSI above 0.2 (watch) or 0.3 (act), read against the largest month-to-month PSI seen in training |
| Alert volume | alerts per AHU per month, per building | 2x the validation rate |
| Outcome | precision from CMMS outcomes, time to close | precision below 60% over 4 weeks |

**Retraining triggers.** Besides the weekly job, a retrain starts early when a key feature crosses PSI 0.3 in more than one building, when precision from CMMS outcomes falls more than 5 points below its level over the previous quarter for 4 weeks, when 10 or more new confirmed labels have arrived, or when a building changes its schedules or setpoints. A retrained bundle still has to pass both promotion gates.

**First month after go-live:** every alert is reviewed by an engineer before it reaches the CMMS and every no-fault-found outcome is reviewed with the technician within 48 h. A rising alert rate with no rise in confirmed faults usually means the building changed (schedule, setpoints, BMS update), so that building is re-baselined rather than the model retuned for everyone.

## 3.5 Onboarding a new building (alerts in under a week when the BMS keeps history)

1. **Day 1:** install the agent and fill the point-mapping template. The agent validates the mapping against the BMS point list and reports any missing points.
2. **Days 1-2:** backfill as much history as the BMS keeps (usually 3 to 12 months). Run the DQ checks on the backfill and review them with the site's engineer. This is where wrong clocks and wrong units turn up.
3. **Days 2-5:** score the building with the fleet models in shadow mode. A new AHU starts from the fleet models, but without its own history it raises about 0.75 false alarms per AHU-month in Q2's test, roughly 60× the shipped rate on known units (0.0125). Refitting the normal models on 14 healthy days brings that to 0.036. The days can come from the backfill. If step 2 yields two weeks that pass the DQ checks, the refit runs on day 2 and the building is ready by day 5. Only a building whose BMS keeps no history waits two weeks.
4. **Go-live** on day 5 with backfilled history (two weeks without), with an engineer reviewing every alert before it reaches the CMMS. If the first two weeks bring more than one unconfirmed alert, the building goes back to shadow mode for a mapping check. On 12 to 16 units, two weeks is only 6 to 8 AHU-months, so this catches a badly mapped building but cannot prove a rate of 0.1 false alarms per AHU-month.
5. **Leaving "new unit" status.** At 0.036 false alarms per AHU-month a refit unit is still about 3× noisier than a known one (0.0125), so go-live is not the end of onboarding. The normal models are refit again at 30 and 60 days, and a building's alerts skip engineer review only once its own rate over the last month is at or below 0.1 per AHU-month with at least 70% of alerts confirmed. I measured the 14-day point only, so how fast the rate falls after that is an assumption the pilot has to check.

Missing sensors are handled by design. The model already knows which buildings have CO2 and airflow, and Q2 shows that losing them barely changes detection (28 found and 22 typed without CO2, 27 and 21 without airflow too, against 27 and 22 with every sensor on the same in-sample thresholds). Damper tickets from a building without CO2 carry a lower-confidence label, since the one missed damper fault was in the mall. The mixed-air temperature matters more. Without it the model raises 8 false alarms instead of 2 (0.033 per AHU-month), because the temperature balance that carries the damper evidence needs it, so a building lacking that point keeps engineer review on damper alerts and the onboarding report names it as the one sensor worth installing. Both results are upper bounds, since the normal-behaviour models were still fitted with these sensors.
