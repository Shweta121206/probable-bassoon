@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

streamlit run app\ui\streamlit_app.py

pause