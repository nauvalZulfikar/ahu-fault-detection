<div class="title">

# AHU Fault Detection and Diagnosis

**Data Scientist (Machine Learning) take-home, VentureSEA**

Nauval Zulfikar, November 2026

</div>

**How to read this report.**
- Q1 covers data quality and exploratory analysis. Q2 is a one-page summary of the model; the full work is in `code/notebooks/02_modelling.ipynb`. Q3 is the production design and Q4 the client reply.
- **Headline result.** Out of fold on January to June, the model finds 27 of 28 labelled faults and gives the right type for 22. Median detection delay is 9 hours. It raises 2 false alarms in 240 AHU-months, and one of the two is a real fault with no work order.
- **Main recommendation.** New units and buildings need a 14-day listening period before their alerts go live.
