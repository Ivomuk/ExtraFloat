@echo off
python scripts\analyze_business_state_evolution.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_business_state_evolution.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
