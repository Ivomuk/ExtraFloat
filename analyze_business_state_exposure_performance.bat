@echo off
python scripts\analyze_business_state_exposure_performance.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_business_state_exposure_performance.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
