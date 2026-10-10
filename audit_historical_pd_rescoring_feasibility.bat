@echo off
python scripts\audit_historical_pd_rescoring_feasibility.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: audit_historical_pd_rescoring_feasibility.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
