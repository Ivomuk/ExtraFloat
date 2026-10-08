@echo off
python scripts\analyze_capacity_level_vs_absolute_exposure.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_capacity_level_vs_absolute_exposure.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
