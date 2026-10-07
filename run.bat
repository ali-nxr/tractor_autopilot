@echo off
REM Tractor Vision V1 - one-click launcher (Windows)
REM Creates a virtual environment on first run, installs dependencies, then runs the dashboard.

cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/checking dependencies...
pip install -r requirements.txt

echo.
echo Starting Tractor Vision V1...
echo (Make sure the RealSense camera is plugged in.)
echo.
python main.py

pause
