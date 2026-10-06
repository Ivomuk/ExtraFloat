@echo off
python scripts\fit_shadow_risk_calibration.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: fit_shadow_risk_calibration.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
