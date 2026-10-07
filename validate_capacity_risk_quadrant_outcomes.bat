@echo off
python scripts\validate_capacity_risk_quadrant_outcomes.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: validate_capacity_risk_quadrant_outcomes.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
