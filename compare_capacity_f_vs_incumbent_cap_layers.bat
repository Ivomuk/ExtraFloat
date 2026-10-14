@echo off
python scripts\compare_capacity_f_vs_incumbent_cap_layers.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_capacity_f_vs_incumbent_cap_layers.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
