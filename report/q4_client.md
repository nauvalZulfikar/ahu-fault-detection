# Question 4: Client communication

## 4.1 Reply to the facilities manager

**Subject:** Last week's 9 alerts, and what happens before the next buildings

Dear [Name],


Thank you for telling us straight. You are right to be frustrated. Four call-outs that found nothing is four wasted trips, and if your team stops trusting the alerts, the system is not doing its job. That is our problem to fix, not your technicians'.

Here is what we will do, starting this week:

1. **We go through all 9 alerts with you.** For each one we will show what the system saw and tell you plainly whether it was our mistake. Some faults, like a temperature sensor that reads 2 °C high, cannot be seen by walking up to the unit. For those we will give your technicians a 10-minute check to confirm or rule it out. The rest we will treat as false alarms and fix.
2. **For the next two weeks, our engineers check every alert before it reaches your team.** Your technicians will only get alerts we are confident about, each with a short note saying what to check and where. We expect that to be one or two a week for this building, not nine.
3. **You get a one-page summary every Friday:** alerts sent, what your team found, and what we changed.

For the building owner, I suggest this message: *the pilot is working but still being tuned; the two new buildings will start next month in "listening mode", where we collect their data and check it ourselves, with no alerts to technicians; alerts go live there once this building has had two clean weeks.* The owner keeps the timeline, and your team gets no extra load until the system has earned it.

Could we have 45 minutes with you and one of the technicians who did the checks, on Tuesday or Wednesday? Their notes on what they found will help us more than anything else.

Kind regards,\
Nauval Zulfikar\
Data Scientist, VentureSEA

## 4.2 Internal note to the project lead

**What I think is happening.** In validation the model raised about one alarm per AHU every seven months, almost all of them real faults. For a building of 12-16 AHUs that is roughly one alert every two weeks, so 9 in one week is about 20 times what we expected. Something is different in this building, not just unlucky. The likely causes, from most to least likely:
- **The building's baseline is probably too short.** If the normal-behaviour models only know this building from its first two weeks, that is too little; in validation every unit had six months, and Q2 shows a unit with no history raises about 20 times more false alarms. A short baseline sees normal swings (a rainy week, a schedule change) as faults.
- **One root cause raised several alerts.** If the central plant's chilled water ran warm, every AHU looks like it has a weak coil. That is one plant ticket, not five AHU tickets.
- **Data problems were read as faults.** In the training data alone we found a logger on the wrong clock, a frozen sensor and a power meter in the wrong units. Any of these in the pilot building would trigger alerts.
- **Some "nothing found" may be real faults** that a visual check cannot see (sensor bias, a valve stuck at 30%). We should not assume all 4 were false.

**Next two weeks.**
- Days 1-3: triage all 9 alerts against the raw trends; label each one *real fault / data issue / plant-level / false alarm*. Check the four "no fault found" units with a reference thermometer and a valve stroke test.
- Days 3-5: add a building-level grouping rule (same signal moving on most AHUs at once means a plant ticket), add DQ gating (no fault alert while a sensor is flagged), and refit the thresholds for this building on everything collected so far.
- Weeks 1-2: human review of every alert before it reaches the CMMS. Track precision and the false alarm rate per AHU-month daily.

**What I need from the client.** Work-order notes for all 9 alerts (what was checked, how); 45 minutes with a technician; the BMS point list and any changes made since go-live; notice of planned maintenance and chiller plant events.

**Rollout recommendation.** Do not go live with alerts in the two new buildings next month. Connect them now and run them in listening mode (data, DQ checks and baselines, no alerts). Turn alerts on when the pilot meets two weeks with at least 70% of alerts confirmed and fewer than 0.1 false alarms per AHU per month. On current evidence that is four to six weeks later than planned, and it means the new buildings start with a month of baseline history instead of two weeks, which fixes the main cause of this problem before it happens again.
