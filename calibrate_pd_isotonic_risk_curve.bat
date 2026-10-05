@echo off
python scripts\calibrate_pd_isotonic_risk_curve.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: calibrate_pd_isotonic_risk_curve.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
