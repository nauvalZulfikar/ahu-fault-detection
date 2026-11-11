# AHU fault detection and diagnosis (code)

## Reproduce `predictions.csv`

```bash
cd code                                # every command below runs from code/
python3.12 -m venv .venv && source .venv/bin/activate   # or uv venv --python 3.12 --seed .venv
pip install -r requirements.txt

# put ahu_train.parquet, ahu_test.parquet, ahu_metadata.csv and fault_labels_train.csv in a data/ folder
# next to code/ (no export needed), or point to them with
# export AHU_DATA_DIR=/abs/path/to/data

python run.py                          # overwrites ../predictions.csv, about 2 to 5 minutes on a 10-core machine and 4 GB RAM (8 threads)
python starter/validate_submission.py ../predictions.csv
```

`run.py` is deterministic, with fixed seeds and LightGBM on CPU. Running it twice gives the same file byte for byte
(`shasum -a 256 ../predictions.csv` gives `f19ade444892bc669efee596b920c0781730718bb02ef778bc49ae67f9aa20ca`).

## Tests

```bash
python -m pytest tests          # 23 tests, 2 to 30 seconds, no data files needed
```

The tests cover every cleaning step (duplicates, units, out-of-range, frozen sensors, clock shift, running flags),
that trailing windows never use future rows or another AHU, the alarm state machine and event scorer, the confidence
intervals and McNemar test, the baselines and the purged month folds.

## Data dictionary, model bundle and experiment log

`DATA_DICTIONARY.md` lists every raw and derived column with unit, typical range, missing share and formula, and keeps
apart the four kinds of "no value" (not fitted, dropout, invalid, frozen). The notebook writes the final model to
`outputs/model/bundle.joblib` (not in git, rebuilt by the notebook) with a readable `outputs/model/bundle.json` that
records data hashes, code commit, library versions, seed and evaluation numbers. Every tuning run, baseline, seed and
experiment is logged in `outputs/experiment_log.csv`.

`starter/validate_submission.py` is the starter validator with one change, it reads the test file from
the same `AHU_DATA_DIR` as the rest of the code.

The report is built from `../report/*.md` with `sh ../report/build_report.sh` (pandoc and Chrome).
The notebooks write their figures into `../report/figures/`.

## Notebook

`notebooks/ahu_fault_detection.ipynb` holds the whole analysis in the order it was done.

| Sections | What they show |
|---|---|
| 0 | Problem, success measures, baselines and what is out of scope |
| 1 to 6 (Q1) | Data dictionary, coverage, every data-quality check and its fix, patterns across buildings and fault signatures vs physics, on January to June only |
| 7 to 15 (Q2) | Framing, residual models, leave-one-month-out validation, threshold tuning, results, comparison with three simpler methods (BMS alarm, expert rules, logistic regression), uncertainty, seeds, nested tuning, Holm correction, calibration and recalibration, error analysis, new-AHU and fewer-sensor experiments, final predictions |
| 16 to 21 | Unlabelled leads in the test period, behaviour tests, the maintenance team's view, model bundle and experiment log, monitoring plan, conclusions |

Run it from `notebooks/` with `jupyter nbconvert --to notebook --execute --inplace ahu_fault_detection.ipynb` (about 20 minutes on a 10-core machine, most of it the new-AHU and missing-sensor runs and the five-seed check).

## Layout

```
run.py                     entry point -> ../predictions.csv
src/ahu_fdd/
  config.py                paths, class names, metadata and label loaders
  clean.py                 data-quality fixes (clock shift, duplicates, units, out-of-range, flatlines), each logged
  physics.py               air-side physics (copy of the starter ahu_utils.py)
  features.py              physics features, fleet normal-behaviour models (residuals), trailing windows
  pipeline.py              leave-one-month-out folds, classifier, end-to-end run()
  events.py                causal alarm state machine and event-level scoring (detection, delay, type, false alarms)
  experiments.py           new-AHU and fewer-sensor experiments
  baseline.py              simpler methods for comparison (BMS supply-air alarm, expert rules, logistic regression)
starter/validate_submission.py
```

## Pipeline in one paragraph

Clean the raw log (`clean.py`) → add physics features per row → fit one LightGBM model of normal
behaviour per signal (valve, fan, mixed air, damper, CO2, return air) on rows with no work order, AHU
as a categorical input, and take residuals → 3 h and 12 h trailing medians of residuals and physics
features → LightGBM multiclass (normal + 4 faults) → per-AHU alarm state machine (open at smoothed
P(fault) > 0.8, close after 12 h below 0.3, type is the class with the highest mean probability since the alarm opened).
Only steady running rows (fan on for at least 30 min) are scored. Other rows carry the AHU's last state.
