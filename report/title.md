<div class="title">

# AHU Fault Detection and Diagnosis

**Data Scientist (Machine Learning) take-home, VentureSEA**

Nauval Zulfikar, November 2026

</div>

**How to read this report.**
- Q1 covers data quality and exploratory analysis. Q2 is a one-page summary of the model; the full work is in `code/notebooks/ahu_fault_detection.ipynb`. Q3 is the production design and Q4 the client reply.
- **Headline result.** Out of fold on January to June, the model finds 27 of 28 labelled faults and gives the right type for 20 to 22 (20 when the alarm thresholds are tuned inside the folds). Median detection delay is 9 to 11 hours. It raises 2 false alarms in 240 AHU-months, and one of the two is a real fault with no work order. The engineering rules a building would use instead find 19 and type 10, with 9 false alarms.
- **Main recommendation.** New units and buildings need a 14-day listening period before their alerts go live.
