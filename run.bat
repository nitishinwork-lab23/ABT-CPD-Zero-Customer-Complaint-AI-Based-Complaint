@echo off
REM One-command start (Windows): double-click run.bat
cd /d "%~dp0"
pip install -r requirements.txt
if not exist data\complaints.db python generate_data.py
streamlit run app.py
