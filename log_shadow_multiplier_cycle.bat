@echo off
python scripts\log_shadow_multiplier_cycle.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: log_shadow_multiplier_cycle.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
