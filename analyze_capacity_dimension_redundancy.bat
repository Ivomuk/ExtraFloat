@echo off
python scripts\analyze_capacity_dimension_redundancy.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_capacity_dimension_redundancy.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
