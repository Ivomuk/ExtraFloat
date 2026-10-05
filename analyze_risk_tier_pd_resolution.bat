@echo off
python scripts\analyze_risk_tier_pd_resolution.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_risk_tier_pd_resolution.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
