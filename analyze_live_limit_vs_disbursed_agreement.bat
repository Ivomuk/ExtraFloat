@echo off
python scripts\analyze_live_limit_vs_disbursed_agreement.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_live_limit_vs_disbursed_agreement.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
