@echo off
python scripts\compare_shadow_vs_live_multiplier.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_shadow_vs_live_multiplier.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
