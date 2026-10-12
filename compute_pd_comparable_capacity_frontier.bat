@echo off
python scripts\compute_pd_comparable_capacity_frontier.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compute_pd_comparable_capacity_frontier.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
