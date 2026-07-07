@echo off
title ChurnSense AI Server
echo ========================================================
echo             Starting ChurnSense AI Server...
echo   Navigate to http://localhost:5000 in your browser
echo ========================================================
python app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Flask server.
    echo Make sure dependencies are installed: pip install -r requirements.txt
    echo Make sure model is trained: python churn_model.py
    pause
)
