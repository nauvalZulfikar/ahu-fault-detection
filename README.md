# AHU fault detection and diagnosis

Take-home for the Data Scientist (Machine Learning) role at VentureSEA: detect and diagnose faults in
air-handling units (AHUs) from 5-minute building-management-system data, 40 AHUs in three buildings.

| | |
|---|---|
| [`report.pdf`](report.pdf) | The report: data quality and EDA (Q1), model summary (Q2), production architecture (Q3), client communication (Q4) |
| [`code/notebooks/01_eda.ipynb`](code/notebooks/01_eda.ipynb) | Exploratory analysis and data-quality checks |
| [`code/notebooks/02_modelling.ipynb`](code/notebooks/02_modelling.ipynb) | Modelling, validation, error analysis, experiments |
| [`code/`](code) | Source package, `run.py`, reproduction steps in [`code/README.md`](code/README.md) |
| `predictions.csv` | Predictions for the held-out period (July and August) |

**Result (out of fold, January to June):** 27 of 28 labelled faults found, 22 with the right type,
median delay 9 hours, 2 false alarms in 240 AHU-months (one of them an unlabelled real fault).

The raw dataset belongs to the assignment and is not included.
