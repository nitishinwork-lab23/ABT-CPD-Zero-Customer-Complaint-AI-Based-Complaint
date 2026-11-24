#!/usr/bin/env bash
# One-command start (Linux/macOS):  ./run.sh
cd "$(dirname "$0")"
pip install -r requirements.txt
[ -f data/complaints.db ] || python generate_data.py
streamlit run app.py
