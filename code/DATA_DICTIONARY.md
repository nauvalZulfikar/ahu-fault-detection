# Data dictionary

**Unit of observation.** One row is one AHU at one 5-minute timestamp (local time, UTC+8). Rows are grouped
by AHU (40 units, about 69,000 rows each) and by building (B1 office with 16 units, B2 hospital with 12, B3 mall with 12).
Rows from the same AHU are strongly correlated in time, so every split in this project is by time (whole months)
or by AHU, never by random row.

**Personal data.** None. The data holds equipment readings and AHU identifiers only. Fault labels come from work
orders but carry no technician names. UU PDP and similar rules therefore do not apply to the modelling data.

**How ranges were computed.** The "typical range" is the 1st to 99th percentile over fan-running rows after
cleaning, January to August 2025 (2,762,088 rows). The "hard limits" are the minimum and maximum after cleaning.
Missing shares are over all rows. Numbers come from `clean.load_all_raw()` and `clean.load_clean()`.

## Raw columns (from the BMS trend log)

| Column | Meaning | Unit | Typical range (running) | Hard limits | Missing (raw / clean) | Source |
|---|---|---|---|---|---|---|
| `timestamp` | Sample time, local | | 5-minute steps | Jan 1 to Aug 31 2025 | 0% | BMS |
| `ahu_id` | Unit identifier, building prefix B1, B2, B3 | | 40 values | | 0% | BMS |
| `fan_status` | Supply fan running | 0 or 1 | 1 | 0 to 1 | 0% (fan off in 35.4% of rows) | BMS |
| `fan_speed_pct` | Variable-speed drive output | % | 35 to 100 | 0 to 100 | 0% | BMS |
| `fan_power_kw` | Fan electrical power | kW | 0.36 to 25.6 | 0.04 to 30.8 | 0.15% / 0.15% | BMS (B3-AHU-04 logged in W from 1 to 21 April, fixed) |
| `supply_static_pa` | Duct static pressure | Pa | 192 to 329 | 0 to 350 | 0% | BMS |
| `supply_static_sp_pa` | Static pressure setpoint | Pa | 220 to 320 | 0 to 320 | 0% | BMS |
| `sat_c` | Supply air temperature after the coil | °C | 12.1 to 18.1 | 6.9 to 32.6 | 0.15% / 0.24% | BMS |
| `sat_sp_c` | Supply air temperature setpoint | °C | 12.5 to 14.5 | 12.5 to 14.5 | 0% | BMS |
| `mat_c` | Mixed air temperature before the coil | °C | 22.1 to 26.2 | 20.9 to 31.9 | 0.15% / 0.15% | BMS |
| `rat_c` | Return air temperature | °C | 21.8 to 25.6 | 20.7 to 31.6 | 0.15% / 0.23% | BMS |
| `zone_temp_sp_c` | Space temperature setpoint | °C | 23 to 25 | 23 to 25 | 0% | BMS |
| `oat_c` | Outdoor air temperature | °C | 23.3 to 32.3 | 20.6 to 33.7 | 0% | BMS (B1 clock shifted 8 h for one week, fixed) |
| `oa_rh_pct` | Outdoor relative humidity | % | 59.7 to 96.1 | 52.6 to 99.0 | 0% | BMS |
| `chw_valve_cmd_pct` | Chilled-water valve command | % | 24.2 to 100 | 0 to 100 | 0.15% / 0.15% | BMS |
| `oa_damper_cmd_pct` | Outdoor-air damper command | % | 15 to 35.3 | 0 to 70 | 0% | BMS |
| `chwst_c` | Chilled-water supply temperature (central plant) | °C | 6.2 to 8.2 | 5.8 to 10.8 | 0% | BMS |
| `filter_dp_pa` | Pressure drop across the filter | Pa | 17 to 210 | 0 to 269 | 0% | BMS |
| `co2_ppm` | Return air CO2 | ppm | 436 to 1,030 | 400 to 2,028 | 30.4% / 30.3% | BMS, B1 and B2 only |
| `sa_flow_m3h` | Supply airflow | m³/h | 2,610 to 12,980 | 2,350 to 16,740 | 70.8% / 70.7% | BMS, B2 only |

## Four kinds of "no value"

The code keeps these apart, because each one means something different to the model and to the client.

| Kind | Where it occurs | How it is stored | Why |
|---|---|---|---|
| **Not applicable** (sensor not fitted) | `co2_ppm` in B3, `sa_flow_m3h` in B1 and B3 | NaN plus the flags `has_co2` and `has_flow` from the metadata | The model must know the sensor is absent, not that the value is unknown |
| **Dropout** (single missing sample) | 0.15% of rows in five signals, January to June only, spread evenly over all 40 units and months, never longer than 2 rows | NaN, not filled | Trailing medians need only a third of the window, so isolated gaps do not matter. Filling would invent readings |
| **Invalid** (physically impossible) | 781 CO2 readings on B2-AHU-03 (20 January to 27 February) at or below 0 or at 5,000 ppm and above | Masked to NaN (`mask_out_of_range`) | A broken reading is not a low or high reading |
| **Frozen** (sensor stuck on one value) | `sat_c` and `rat_c` for 36 or more rows in a row (B2-AHU-08 in July with 2,587 rows, B1-AHU-07 in May with 2,239 rows) | Masked to NaN (`mask_flatlines`), row kept | A frozen sensor would look like a perfect controller. The row itself is still valid for every other signal |
| **True zero** | `chw_valve_cmd_pct` at 0% in 0.18% of running rows, every signal while the fan is off | Kept as 0 | A closed valve is a real command, not a missing one |
| **Logging gap** (no row at all) | B2, one gap of 8 days (10 to 18 March) on every unit, plus short gaps elsewhere (train 98.5% and test 99.7% complete) | No rows created | Inventing rows would invent data. Windows count rows, so a gap shortens the window rather than filling it |

## Metadata (`ahu_metadata.csv`, one row per AHU)

| Column | Meaning | Unit |
|---|---|---|
| `building`, `building_type` | B1 office, B2 hospital, B3 mall | |
| `serves` | Area served (office floors, wards, food court and so on) | |
| `design_airflow_m3h` | Design supply airflow | m³/h |
| `fan_rated_kw` | Fan motor rating | kW |
| `has_co2_sensor`, `has_airflow_station` | Sensors fitted | True or False |
| `operating_hours` | Schedule as text | |

## Labels (`fault_labels_train.csv`)

`ahu_id`, `fault_type` (four types, see the brief), `start`, `end`. There are 28 events in January to June. Start is the
engineer's estimate and end is the repair time, so rows within 1 day of an event are treated as uncertain and
not used as "normal" training rows. Labels are known to be incomplete (notebook 02 finds one unlabelled stuck damper).

## Derived columns (`features.add_physics`, `clean.add_running_flags`)

| Column | Formula | Unit | Meaning |
|---|---|---|---|
| `min_since_start` | minutes since the fan last switched on | min | Start-up transient |
| `steady` | `fan_status == 1` and `min_since_start >= 30` | bool | Rows the model learns from and scores |
| `sat_err` | `sat_c - sat_sp_c` | °C | Is the coil holding its setpoint |
| `coil_dt` | `mat_c - sat_c` | °C | Cooling done by the coil |
| `mix_dt` | `mat_c - rat_c` | °C | How much outdoor air changes the mix |
| `oaf` | `(mat_c - rat_c) / (oat_c - rat_c)`, NaN when the denominator is under 2 °C, clipped to -0.5 to 1.5 | fraction | Outdoor-air fraction from the mixing balance |
| `static_err` | `supply_static_pa - supply_static_sp_pa` | Pa | Is the fan holding duct pressure |
| `approach` | `sat_c - chwst_c` | °C | Coil approach temperature |
| `power_ratio` | `fan_power_kw / (fan_rated_kw × (max(fan_speed_pct, 20) / 100) ^ 2.5)` | ratio | Fan power against the affinity law |
| `flow_est` | `sa_flow_m3h`, else `design_airflow_m3h × fan_speed_pct / 100` | m³/h | Airflow, measured in B2 and estimated elsewhere |
| `cool_kw` | `1.207 × flow_est / 3600 × coil_dt` | kW | Sensible cooling delivered |
| `hour`, `dow`, `building_code` | from `timestamp` and `ahu_id` | | Schedule and building |
| `r_valve`, `r_fan`, `r_mat`, `r_damper`, `r_co2`, `r_rat` | measured minus predicted by the fleet normal-behaviour model | signal units | Fault evidence (`features.NormalModels`) |
| `*_med3h`, `*_med12h` | trailing median over the last 36 or 144 steady rows of the same AHU | signal units | Persistence, backwards in time only |
| `valve_hi_*`, `valve_lo_*`, `sat_off_*` | share of the window with valve above 95%, below 5%, or `abs(sat_err) > 1.5` | fraction | Saturation and setpoint loss |
| `valve_coil_corr_12h` | trailing correlation of valve command and `coil_dt` over 144 rows | | Does the coil respond to the valve |
| `r_valve_trend`, `r_fan_trend` | 3 h median minus 12 h median | signal units | Is the residual growing |
