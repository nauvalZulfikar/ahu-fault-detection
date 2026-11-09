# AHU fault detection and diagnosis: code

## Reproduce `predictions.csv`

```bash
# Python 3.12
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the four data files from the assignment package
export AHU_DATA_DIR=/path/to/data      # default: ../data, i.e. a data/ folder next to code/

python run.py                          # writes ../predictions.csv (about 2 minutes, 8 cores)
python starter/validate_submission.py ../predictions.csv
```

`run.py` is deterministic: fixed seeds, and LightGBM runs on CPU.

`starter/validate_submission.py` is the starter validator with one change: it reads the test file from
the same `AHU_DATA_DIR` as the rest of the code.

The report is built from `../report/*.md` with `sh ../report/build_report.sh` (pandoc and Chrome).
The notebooks write their figures into `../report/figures/`.

## Notebooks

| Notebook | What it shows |
|---|---|
| `notebooks/01_eda.ipynb` | Q1: coverage, every data-quality check and its fix, patterns across buildings, fault signatures vs physics, unlabelled anomalies |
| `notebooks/02_modelling.ipynb` | Q2: framing, residual models, leave-one-month-out validation, threshold tuning, results, error analysis, new-AHU and fewer-sensor experiments, final predictions |

Run them from `notebooks/`: `jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb` (02 takes about 10 minutes).

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
starter/validate_submission.py
```

## Pipeline in one paragraph

Clean the raw log (`clean.py`) → add physics features per row → fit one LightGBM model of normal
behaviour per signal (valve, fan, mixed air, damper, CO2, return air) on rows with no work order, AHU
as a categorical input, and take residuals → 3 h and 12 h trailing medians of residuals and physics
features → LightGBM multiclass (normal + 4 faults) → per-AHU alarm state machine (open at smoothed
P(fault) > 0.8, close after 12 h below 0.3, type = highest mean class probability since the alarm opened).
Only steady running rows (fan on for at least 30 min) are scored. Other rows carry the AHU's last state.
