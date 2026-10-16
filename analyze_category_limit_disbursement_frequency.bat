@echo off
python scripts\analyze_category_limit_disbursement_frequency.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_category_limit_disbursement_frequency.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
