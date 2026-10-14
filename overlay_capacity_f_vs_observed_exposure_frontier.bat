@echo off
python scripts\overlay_capacity_f_vs_observed_exposure_frontier.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: overlay_capacity_f_vs_observed_exposure_frontier.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
