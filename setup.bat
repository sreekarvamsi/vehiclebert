@echo off
REM VehicleBERT Quick Setup Script for Windows
REM This script sets up the VehicleBERT project from scratch

echo ==========================================
echo   VehicleBERT Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
echo [OK] Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] Pip upgraded
echo.

REM Install requirements
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed
echo.

REM Install in development mode
echo Installing VehicleBERT package...
pip install -e . --quiet
echo [OK] Package installed
echo.

REM Create necessary directories
echo Creating directories...
if not exist "data\annotated" mkdir data\annotated
if not exist "data\raw" mkdir data\raw
if not exist "models" mkdir models
echo [OK] Directories created
echo.

REM Generate sample data
echo Generating sample dataset...
echo Creating 1,000 sample sentences for quick testing...
python scripts\prepare_data.py --num-sentences 1000
echo [OK] Sample data generated
echo.

REM Summary
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo Quick Start Commands:
echo.
echo 1. Train a model (quick test - 3 epochs):
echo    python scripts\train.py --epochs 3 --batch-size 8
echo.
echo 2. Run inference:
echo    python scripts\inference.py --model-path models\vehiclebert\best --text "Replace the O2 sensor"
echo.
echo 3. Evaluate model:
echo    python scripts\evaluate.py --model-path models\vehiclebert\best --test-file data\test.json
echo.
echo 4. Open Jupyter notebook:
echo    jupyter notebook notebooks\demo_notebook.ipynb
echo.
echo Documentation:
echo   - README.md        - Full documentation
echo   - QUICKSTART.md    - Quick start guide
echo   - GITHUB_SETUP.md  - GitHub setup instructions
echo   - DEPLOYMENT.md    - Deployment guide
echo.
echo For production training with 5,000 sentences:
echo    python scripts\prepare_data.py --num-sentences 5000
echo    python scripts\train.py --epochs 10
echo.
echo Happy coding! (car emoji)
pause
