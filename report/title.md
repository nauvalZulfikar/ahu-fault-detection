<div class="title">

# AHU Fault Detection and Diagnosis

**Data Scientist (Machine Learning) take-home, VentureSEA**

Nauval Zulfikar, November 2026

</div>

**How to read this report.**
- Q1 covers data quality and exploratory analysis. Q2 is a one-page summary of the model; the full work is in `code/notebooks/ahu_fault_detection.ipynb`. Q3 is the production design and Q4 the client reply.
- **Headline result.** Out of fold on January to June, the model as shipped finds 27 of 28 labelled faults and gives the right type for 21, with alarm thresholds tuned inside the folds. Median detection delay is 10 hours. It raises 3 false alarms in 240 AHU-months, and one of them looks like a real stuck damper with no work order. The engineering rules a building would use instead find 19 and type 10, with 9 false alarms.
- **Main recommendation.** New units and buildings need a 14-day listening period before their alerts go live.
