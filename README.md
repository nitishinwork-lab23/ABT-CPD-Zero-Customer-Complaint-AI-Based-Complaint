# ZeroC-AI – Zero Customer Complaint Analytics & Prediction System (PS-036 prototype)

AI-based complaint analytics, root-cause identification and prediction for ABT-CPD.
Runs on a normal laptop (no GPU, no internet needed after install).

## Run it (2 minutes)
```bash
pip install -r requirements.txt
python generate_data.py        # creates data/complaints.db (synthetic 19-year demo data) – skip if it already exists
streamlit run app.py           # opens http://localhost:8501
```
Windows: double-click `run.bat`.  Linux/macOS: `./run.sh`.  Self-test: `python test_smoke.py`.

## What each page shows (mapped to the 9 deliverables)
| # | Deliverable | Where in the app | How |
|---|---|---|---|
| 1 | Classify complaints | Pareto & Classification | TF-IDF + Logistic Regression on complaint text (9 categories), k-means theme discovery |
| 2 | Pareto / major contributors | Pareto & Classification | 80-20 Pareto by category, product, customer, line, root cause |
| 3 | Recurring / chronic problems | Chronic & Recurring | rolling 24-month counts, trend, action-effectiveness; Chronic ≥10, Recurring 5–9 |
| 4 | Product–customer–type–root-cause links | Relationships & Root Cause | lift matrices, association rules, 6M root-cause shares, process-parameter box-plots |
| 5 | High-risk products / processes | Risk & Prediction | risk league table + product ranking |
| 6 | Predict complaint risk | Risk & Prediction | gradient boosting; P(complaint next month) per product × category; time-based back-test, calibration, importance |
| 7 | Preventive / corrective actions | Preventive Recommendations + alerts | similar-case retrieval, effectiveness-weighted ranking, SOP rule checks |
| 8 | Management dashboard + alerts | Command Centre | KPI tiles, ranked alerts (model risk + SPC u-chart/CUSUM), trend, heat-map |
| 9 | KPIs towards zero | KPIs: Road to Zero | complaints per million units, repeat rate, CAPA effectiveness, closure time, early-warning precision, chronic count, complaint-free pairs |
| – | New data & continuous learning | Log New Complaint | auto-classifies text, saves to SQLite; “Retrain models now” refreshes everything |

## Using the REAL ABT-CPD data
Edit `COLUMN_MAP` in `load_real_data.py`, then `python load_real_data.py your_file.xlsx [units_dispatched.csv]`, restart the app and press **Retrain models now**.
Mandatory columns: date, customer, product, complaint_text, category. Everything else (root cause, corrective action, effectiveness Y/N, batch, line, process values, test result, closure days, severity) is optional – the system works with whatever exists.
Remember to change the “SYNTHETIC DEMO” tag in `app.py` (search `ZC_DATA`) – set the environment variable `ZC_DATA=real`.

## Honest notes for the judges' Q&A
* The bundled data is **synthetic** with planted patterns (a chronic foreign-matter problem on Product B, a rising packaging trend on Product D, seasonal infestation on Product E, …). It proves the pipeline end-to-end; **quote accuracy only after re-running on real data.**
* The risk model is back-tested on unseen months (train ≤ Dec 2023, test Jan 2024 → Oct 2026) and compared with a simple “last 6 months” baseline – on the real data this table is the evidence.
* Predictions are risk indicators with plain-language drivers, not proof of causation; QA confirms root cause.

## Files
`app.py` dashboard · `engine.py` analytics/ML · `generate_data.py` synthetic data · `load_real_data.py` import · `test_smoke.py` tests · `data/complaints.db` SQLite database
