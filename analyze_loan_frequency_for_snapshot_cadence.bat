@echo off
python scripts\analyze_loan_frequency_for_snapshot_cadence.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_loan_frequency_for_snapshot_cadence.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
