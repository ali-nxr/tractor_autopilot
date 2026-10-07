#!/usr/bin/env bash
# Tractor Vision V1 - one-click launcher (Linux/Mac)
# Creates a virtual environment on first run, installs dependencies, then runs the dashboard.

set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing/checking dependencies..."
pip install -r requirements.txt

echo
echo "Starting Tractor Vision V1..."
echo "(Make sure the RealSense camera is plugged in.)"
echo
python main.py
