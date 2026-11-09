# Question 3: Production architecture

![Architecture](figures/architecture.png){style="max-height:none"}

## 3.1 Data path

**Edge (one small agent per building).** A Windows service reads new rows from the BMS SQL Server every 5 minutes. It uses the timestamp plus the row key as a watermark, so each read is incremental. It maps the site's point names to our standard schema using a per-building mapping file (a YAML file kept in Git), writes compressed Parquet into a local buffer, and pushes the files over outbound HTTPS to an ingest API. The agent does no modelling. Because the PC only allows outbound traffic, the agent also polls the API for config and version updates, so we never need inbound access. The buffer holds 14 days, so an internet outage loses nothing: the agent re-sends when it reconnects and the ingest API ignores duplicates by (ahu_id, timestamp, hash). The agent sends a heartbeat every 5 minutes.

**Cloud storage.** Files land in object storage in two zones:
- *raw*: append-only and never edited, so any result can be replayed.
- *curated*: the output of the same `clean.py` used in this take-home, run as data-quality (DQ) checks. These cover clock shifts, duplicate exports, unit errors, out-of-range values, frozen sensors and gaps. Each check writes a DQ flag per AHU per hour instead of silently fixing data. Flags feed both the model, which gives no fault alert while a key sensor is flagged, and a DQ ticket queue, because a frozen sensor is itself worth a work order.

**Scale.** 600 AHUs at 12 rows an hour is about 173k rows a day, or about 2,000 AHUs and 576k rows a day at the stated target. That is a few hundred MB a day in Parquet. It fits comfortably in one Postgres/TimescaleDB instance or DuckDB over Parquet. We do not need Spark or streaming: an hourly batch job on one machine handles the whole fleet in minutes.

## 3.2 Models

**Training (weekly, or on demand).** The job retrains the fleet normal-behaviour models and the fault classifier on all curated history, with no-fault rows defined as rows without an open work order. It runs the same leave-one-month-out evaluation as the notebook. The output is one versioned bundle: model files, alarm thresholds, feature list, code commit and evaluation report, stored in a model registry (MLflow or a versioned bucket).

**Promotion.** A new bundle runs in *shadow mode* for at least a week. It scores live data and logs alerts, but sends nothing. It replaces the live bundle only if, on the same week, its confirmed alerts are no fewer and its false-alarm rate is no higher. Rollback means pointing back to the previous bundle ID.

**Inference (hourly).** For each AHU the job takes the last 24 h of curated data, computes features, scores, and runs the alarm state machine. Its state (open alarms, cumulative type probabilities) persists between runs, so the logic is exactly the causal code validated offline. Hourly is enough because these faults develop over hours and cost money over days: the validated median delay is about 10 hours, set by the 3-12 h evidence windows, not by how often we score. Scoring every 5 minutes would add cost and alert churn and find nothing sooner. Safety-critical conditions (a fan trip, a fire damper) stay with the BMS's own alarms.

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
| Input drift | residual distributions per AHU vs the training window | median residual shifts beyond its normal band for a week |
| Alert volume | alerts per AHU per month, per building | 2x the validation rate |
| Outcome | precision from CMMS outcomes, time to close | precision below 60% over 4 weeks |

**First month after go-live:** every alert is reviewed by an engineer before it reaches the CMMS, every no-fault-found outcome is reviewed with the technician within 48 h, and these numbers go to the client weekly. A rising alert rate with no change in confirmed faults usually means a building has changed (new schedule, setpoints, BMS update), not that the units broke. The response is to re-baseline that building, not to retune the model for everyone.

## 3.5 Onboarding a new building (connected in under a week, alerts after two weeks)

1. **Day 1:** install the agent and fill the point-mapping template. The agent validates the mapping against the BMS point list and reports any missing points.
2. **Days 1-2:** backfill as much history as the BMS keeps (usually 3 to 12 months). Run the DQ checks on the backfill and review them with the site's engineer. This is where wrong clocks and wrong units turn up.
3. **Days 2-5:** score the building with the fleet models in shadow mode. A new AHU starts from the fleet models, which already know buildings of the same type. Without its own history, though, it raises about 0.75 false alarms per AHU-month in Q2's test, roughly 20× the normal rate. So the normal models are refit once the unit has 14 healthy days. In Q2 that brings the rate down to 0.036 per AHU-month.
4. **Go-live** only after the building has had two weeks in shadow mode with an acceptable alert rate (below 0.1 false alarms per AHU-month). Until then, an engineer reviews every alert before it reaches the CMMS.

Missing sensors are handled by design. The feature set already distinguishes buildings with and without CO2 and airflow, and Q2 shows how much detection drops without them. A building with fewer points simply gets the reduced model and a lower confidence label on its tickets.
