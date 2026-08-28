@echo off
setlocal
cd /d "%~dp0"

python -c "import streamlit, plotly" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Missing dependencies. Run: python -m pip install -r requirements.txt
  exit /b 1
)

netstat -ano -p tcp | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 (
  echo [ERROR] Port 8501 is already in use. Stop the existing local process or choose another port before starting the demonstration.
  exit /b 2
)

echo Starting HDC System Demonstration at http://127.0.0.1:8501
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
