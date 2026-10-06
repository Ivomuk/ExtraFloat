@echo off
python scripts\check_live_shadow_vs_category_limit.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_live_shadow_vs_category_limit.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
