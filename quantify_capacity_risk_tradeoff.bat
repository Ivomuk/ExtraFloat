@echo off
python scripts\quantify_capacity_risk_tradeoff.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: quantify_capacity_risk_tradeoff.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
