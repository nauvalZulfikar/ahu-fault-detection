# AHU fault detection and diagnosis

Take-home for the Data Scientist (Machine Learning) role at VentureSEA. The task is to detect and diagnose faults in
air-handling units (AHUs) from 5-minute building-management-system data, 40 AHUs in three buildings.

| | |
|---|---|
| [`report.pdf`](report.pdf) | The report, covering data quality and EDA (Q1), model summary (Q2), production architecture (Q3), client communication (Q4) |
| [`code/notebooks/ahu_fault_detection.ipynb`](code/notebooks/ahu_fault_detection.ipynb) | The whole analysis in one notebook, from problem and data quality to model, validation, error analysis, monitoring and conclusions |
| [`code/`](code) | Source package, `run.py`, reproduction steps in [`code/README.md`](code/README.md) (Python 3.12, about 4 GB RAM, run from `code/`) |
| `predictions.csv` | Predictions for the held-out period (July and August) |

**Result (out of fold, January to June).** The model as shipped finds 27 of 28 labelled faults and names 21 of them
correctly, with the alarm thresholds tuned inside the folds (22 when they are tuned on the scored months). Median
delay is 10 hours, with 3 false alarms in 240 AHU-months (one of them looks like an unlabelled stuck damper).

The raw dataset belongs to the assignment and is not included.
