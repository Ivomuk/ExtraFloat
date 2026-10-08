@echo off
python scripts\analyze_capacity_exposure_performance_matrix.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_capacity_exposure_performance_matrix.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
