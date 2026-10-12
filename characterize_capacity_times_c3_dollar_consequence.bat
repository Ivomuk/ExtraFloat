@echo off
python scripts\characterize_capacity_times_c3_dollar_consequence.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: characterize_capacity_times_c3_dollar_consequence.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
